#!/usr/bin/env bash
# fetch_bal.sh — download BAL datasets (Agarwal et al., ECCV 2010)
# https://grail.cs.washington.edu/projects/bal/
#
# Usage:
#   data/fetch_bal.sh smoke            # smallest problem only (ladybug 49 cams)
#   data/fetch_bal.sh ladybug          # one whole dataset
#   data/fetch_bal.sh ladybug dubrovnik
#   data/fetch_bal.sh all
#   data/fetch_bal.sh <url-to-.bz2>    # one specific file
#
# Files land in data/<dataset>/, kept compressed (.bz2). The BAL reader
# streams .bz2 directly via bzcat, so no manual decompression is needed.

set -euo pipefail

BASE="https://grail.cs.washington.edu/projects/bal"
DATASETS=(ladybug trafalgar dubrovnik venice final)
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fetch_dataset() {
    local ds="$1"
    local urls
    urls=$(curl -fsSL "$BASE/$ds.html" | grep -oE 'href=data/[^ >]*\.bz2' | sed 's/href=//')
    if [[ -z "$urls" ]]; then
        echo "error: no .bz2 links found at $BASE/$ds.html" >&2
        return 1
    fi
    mkdir -p "$DATA_DIR/$ds"
    local n=0
    while IFS= read -r rel; do
        local out="$DATA_DIR/$ds/$(basename "$rel")"
        if [[ -f "$out" ]]; then
            echo "skip (exists): $out"
        else
            echo "fetch: $BASE/$rel"
            curl -fSL --retry 3 -o "$out" "$BASE/$rel"
        fi
        n=$((n + 1))
    done <<< "$urls"
    echo "$ds: $n files"
}

if [[ $# -lt 1 ]]; then
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        smoke)
            mkdir -p "$DATA_DIR/ladybug"
            url="$BASE/data/ladybug/problem-49-7776-pre.txt.bz2"
            out="$DATA_DIR/ladybug/problem-49-7776-pre.txt.bz2"
            [[ -f "$out" ]] || { echo "fetch: $url"; curl -fSL --retry 3 -o "$out" "$url"; }
            echo "smoke: $out"
            ;;
        all)
            for ds in "${DATASETS[@]}"; do fetch_dataset "$ds"; done
            ;;
        http*://*.bz2)
            out="$DATA_DIR/$(basename "$arg")"
            [[ -f "$out" ]] || { echo "fetch: $arg"; curl -fSL --retry 3 -o "$out" "$arg"; }
            ;;
        *)
            found=0
            for ds in "${DATASETS[@]}"; do
                [[ "$arg" == "$ds" ]] && { fetch_dataset "$ds"; found=1; break; }
            done
            [[ $found -eq 0 ]] && { echo "error: unknown dataset '$arg'" >&2; exit 1; }
            ;;
    esac
done
