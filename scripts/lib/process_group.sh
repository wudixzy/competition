#!/bin/bash

# Process-group helpers for benchmark service isolation. A zombie cannot hold a
# GPU or port and cannot be reaped after it has been adopted by PID 1.

bi100_validate_pid() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

bi100_process_group_count() {
    local pgid=$1
    local state=$2
    local table
    bi100_validate_pid "$pgid" || return 2
    [[ "$state" == live || "$state" == zombie ]] || return 2
    table=$(ps -eo pid=,pgid=,stat= 2>/dev/null) || return 2
    awk -v pgid="$pgid" -v state="$state" '
        $2 == pgid {
            zombie = substr($3, 1, 1) == "Z"
            if ((state == "zombie" && zombie) ||
                    (state == "live" && !zombie)) {
                count++
            }
        }
        END { print count + 0 }
    ' <<< "$table"
}

bi100_validate_process_group_leader() {
    local pgid=$1
    local leader_pid=$2
    local table
    local identity
    local observed_pgid
    local state
    bi100_validate_pid "$pgid" || return 2
    bi100_validate_pid "$leader_pid" || return 2
    table=$(ps -eo pid=,pgid=,stat= 2>/dev/null) || return 2
    identity=$(awk -v pid="$leader_pid" '$1 == pid { print $2, $3; exit }' \
        <<< "$table")
    [[ -z "$identity" ]] && return 0
    read -r observed_pgid state <<< "$identity"
    [[ "$state" == Z* ]] && return 0
    if [[ "$observed_pgid" != "$pgid" ]]; then
        echo "service leader $leader_pid moved from process group $pgid to $observed_pgid" >&2
        return 1
    fi
}

bi100_validate_process_group_identity() {
    local pgid=$1
    local leader_pid=$2
    local expected_starttime=$3
    local expected_token=$4

    bi100_validate_pid "$pgid" || return 2
    bi100_validate_pid "$leader_pid" || return 2
    [[ "$expected_starttime" =~ ^[1-9][0-9]*$ ]] || return 2
    [[ "$expected_token" =~ ^[0-9a-f]{32}$ ]] || return 2

    python3 - "$pgid" "$leader_pid" "$expected_starttime" \
            "$expected_token" <<'PY'
from pathlib import Path
import sys

pgid = int(sys.argv[1])
leader_pid = int(sys.argv[2])
expected_starttime = int(sys.argv[3])
expected_token = (
    f"BI100_PROCESS_SESSION_TOKEN={sys.argv[4]}".encode("ascii")
)


def read_stat(path):
    value = path.read_text(encoding="ascii")
    closing = value.rfind(")")
    if closing < 0:
        raise ValueError("malformed process stat")
    fields = value[closing + 2:].split()
    return {
        "state": fields[0],
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime": int(fields[19]),
    }


members = []
for process in Path("/proc").iterdir():
    if not process.name.isdigit():
        continue
    try:
        row = read_stat(process / "stat")
    except (FileNotFoundError, ProcessLookupError):
        continue
    if row["pgid"] == pgid and row["state"] != "Z":
        members.append((int(process.name), process, row))

if not members:
    raise SystemExit(0)

for pid, process, row in members:
    if row["sid"] != pgid:
        print(
            f"process {pid} has unexpected session identity",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if pid == leader_pid and row["starttime"] != expected_starttime:
        print("service leader starttime differs", file=sys.stderr)
        raise SystemExit(1)
    try:
        environment = (process / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        print(f"process {pid} identity disappeared", file=sys.stderr)
        raise SystemExit(1)
    except OSError as exc:
        print(
            f"cannot inspect process {pid} environment: {type(exc).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if expected_token not in environment:
        print(f"process {pid} session token differs", file=sys.stderr)
        raise SystemExit(1)
PY
}

bi100_signal_verified_process_group() {
    local pgid=$1
    local leader_pid=$2
    local expected_starttime=$3
    local expected_token=$4
    local signal_name=$5

    bi100_validate_pid "$pgid" || return 2
    bi100_validate_pid "$leader_pid" || return 2
    [[ "$expected_starttime" =~ ^[1-9][0-9]*$ ]] || return 2
    [[ "$expected_token" =~ ^[0-9a-f]{32}$ ]] || return 2
    [[ "$signal_name" == TERM || "$signal_name" == KILL ]] || return 2

    python3 - "$pgid" "$leader_pid" "$expected_starttime" \
            "$expected_token" "$signal_name" <<'PY'
from pathlib import Path
import ctypes
import errno
import os
import signal
import sys

pgid = int(sys.argv[1])
leader_pid = int(sys.argv[2])
expected_starttime = int(sys.argv[3])
expected_token = (
    f"BI100_PROCESS_SESSION_TOKEN={sys.argv[4]}".encode("ascii")
)
signal_value = getattr(signal, f"SIG{sys.argv[5]}")


def open_pidfd(pid):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid, 0)
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.syscall(434, pid, 0)
    if descriptor >= 0:
        return descriptor
    error = ctypes.get_errno()
    if error == errno.ESRCH:
        raise ProcessLookupError(error, os.strerror(error))
    raise OSError(error, os.strerror(error))


def read_stat(path):
    value = path.read_text(encoding="ascii")
    closing = value.rfind(")")
    if closing < 0:
        raise ValueError("malformed process stat")
    fields = value[closing + 2:].split()
    return {
        "state": fields[0],
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime": int(fields[19]),
    }


pidfd = None
leader_path = Path("/proc") / str(leader_pid)
try:
    try:
        pidfd = open_pidfd(leader_pid)
    except ProcessLookupError:
        pidfd = None
    if pidfd is not None:
        leader = read_stat(leader_path / "stat")
        if leader["starttime"] != expected_starttime:
            print("service leader starttime differs", file=sys.stderr)
            raise SystemExit(1)

    members = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            row = read_stat(process / "stat")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if row["pgid"] == pgid and row["state"] != "Z":
            members.append((int(process.name), process, row))
    if not members:
        raise SystemExit(0)
    for pid, process, row in members:
        if row["sid"] != pgid:
            print(
                f"process {pid} has unexpected session identity",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if pid == leader_pid and row["starttime"] != expected_starttime:
            print("service leader starttime differs", file=sys.stderr)
            raise SystemExit(1)
        try:
            environment = (process / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            print(f"process {pid} identity disappeared", file=sys.stderr)
            raise SystemExit(1)
        except OSError as exc:
            print(
                f"cannot inspect process {pid} environment: "
                f"{type(exc).__name__}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if expected_token not in environment:
            print(f"process {pid} session token differs", file=sys.stderr)
            raise SystemExit(1)
    os.killpg(pgid, signal_value)
except ProcessLookupError:
    pass
finally:
    if pidfd is not None:
        os.close(pidfd)
PY
}

bi100_process_group_snapshot() {
    local pgid=$1
    local table
    bi100_validate_pid "$pgid" || return 2
    table=$(ps -eo pid=,ppid=,pgid=,stat=,comm= 2>/dev/null) || return 2
    awk -v pgid="$pgid" '$3 == pgid { print }' <<< "$table"
}

bi100_wait_for_process_group_quiescent() {
    local pgid=$1
    local attempts=$2
    local live_count
    bi100_validate_pid "$pgid" || return 2
    [[ "$attempts" =~ ^[1-9][0-9]*$ ]] || return 2
    for _ in $(seq 1 "$attempts"); do
        live_count=$(bi100_process_group_count "$pgid" live) || return 2
        ((live_count == 0)) && return 0
        sleep 1
    done
    return 1
}

bi100_stop_process_group() {
    local pgid=$1
    local leader_pid=${2:-}
    local term_attempts=${3:-120}
    local kill_attempts=${4:-20}
    local expected_starttime=${5:-}
    local expected_token=${6:-}
    local live_count
    local wait_rc
    local zombie_count

    bi100_validate_pid "$pgid" || return 2
    if [[ -n "$expected_starttime" || -n "$expected_token" ]]; then
        [[ -n "$leader_pid" && -n "$expected_starttime" \
            && -n "$expected_token" ]] || return 2
        bi100_validate_process_group_identity \
            "$pgid" "$leader_pid" "$expected_starttime" \
            "$expected_token" || return $?
    fi
    if [[ -n "$leader_pid" ]]; then
        bi100_validate_pid "$leader_pid" || return 2
        bi100_validate_process_group_leader \
            "$pgid" "$leader_pid" || return $?
    fi

    live_count=$(bi100_process_group_count "$pgid" live) || return 2
    if ((live_count > 0)); then
        if [[ -n "$expected_starttime" ]]; then
            bi100_signal_verified_process_group \
                "$pgid" "$leader_pid" "$expected_starttime" \
                "$expected_token" TERM || return $?
        else
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
        if bi100_wait_for_process_group_quiescent "$pgid" "$term_attempts"; then
            wait_rc=0
        else
            wait_rc=$?
        fi
        if ((wait_rc == 2)); then
            echo "cannot inspect service process group $pgid" >&2
            return 2
        fi
        if ((wait_rc != 0)); then
            if [[ -n "$expected_starttime" ]]; then
                bi100_signal_verified_process_group \
                    "$pgid" "$leader_pid" "$expected_starttime" \
                    "$expected_token" KILL || return $?
            else
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
            if bi100_wait_for_process_group_quiescent \
                    "$pgid" "$kill_attempts"; then
                wait_rc=0
            else
                wait_rc=$?
            fi
            if ((wait_rc != 0)); then
                echo "service process group $pgid has live members after cleanup" >&2
                bi100_process_group_snapshot "$pgid" >&2 || true
                return 1
            fi
        fi
    fi

    if [[ -n "$leader_pid" ]]; then
        if [[ -n "$expected_starttime" ]]; then
            bi100_validate_process_group_identity \
                "$pgid" "$leader_pid" "$expected_starttime" \
                "$expected_token" || return $?
        else
            bi100_validate_process_group_leader \
                "$pgid" "$leader_pid" || return $?
        fi
        wait "$leader_pid" 2>/dev/null || true
    fi
    live_count=$(bi100_process_group_count "$pgid" live) || return 2
    if ((live_count > 0)); then
        echo "service process group $pgid has live members after cleanup" >&2
        bi100_process_group_snapshot "$pgid" >&2 || true
        return 1
    fi
    zombie_count=$(bi100_process_group_count "$pgid" zombie) || return 2
    if ((zombie_count > 0)); then
        echo "service process group $pgid is zombie-only; zombie_count=$zombie_count" >&2
        bi100_process_group_snapshot "$pgid" >&2 || true
    fi
}
