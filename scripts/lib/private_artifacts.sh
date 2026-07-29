#!/bin/bash

remove_teacher_forced_observations() {
    if [[ $# -ne 1 || -z "$1" ]]; then
        return 2
    fi
    local run_root=$1
    local fixed=(
        "$run_root/control/teacher_forced_observation.json"
        "$run_root/candidate/teacher_forced_observation.json"
    )
    local temporary=()
    local path
    local cleanup_rc=0
    local nullglob_was_set=0

    if shopt -q nullglob; then
        nullglob_was_set=1
    fi
    shopt -s nullglob
    temporary=(
        "$run_root/control"/.teacher_forced_observation.json.*.tmp
        "$run_root/candidate"/.teacher_forced_observation.json.*.tmp
    )
    rm -f -- "${fixed[@]}" "${temporary[@]}"

    for path in "${fixed[@]}"; do
        if [[ -e "$path" || -L "$path" ]]; then
            cleanup_rc=1
        fi
    done
    temporary=(
        "$run_root/control"/.teacher_forced_observation.json.*.tmp
        "$run_root/candidate"/.teacher_forced_observation.json.*.tmp
    )
    if [[ ${#temporary[@]} -ne 0 ]]; then
        cleanup_rc=1
    fi
    if [[ $nullglob_was_set -eq 0 ]]; then
        shopt -u nullglob
    fi
    return "$cleanup_rc"
}
