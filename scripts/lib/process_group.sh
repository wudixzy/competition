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
    table=$(ps -eo pgid=,stat= 2>/dev/null) || return 2
    awk -v pgid="$pgid" -v state="$state" '
        $1 == pgid {
            zombie = substr($2, 1, 1) == "Z"
            if ((state == "zombie" && zombie) ||
                    (state == "live" && !zombie)) {
                count++
            }
        }
        END { print count + 0 }
    ' <<< "$table"
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
    local live_count
    local wait_rc
    local zombie_count

    bi100_validate_pid "$pgid" || return 2
    if [[ -n "$leader_pid" ]]; then
        bi100_validate_pid "$leader_pid" || return 2
    fi

    live_count=$(bi100_process_group_count "$pgid" live) || return 2
    if ((live_count > 0)); then
        kill -TERM -- "-$pgid" 2>/dev/null || true
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
            kill -KILL -- "-$pgid" 2>/dev/null || true
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
