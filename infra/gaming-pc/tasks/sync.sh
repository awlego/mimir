#!/usr/bin/env bash
# Fast-forward the WSL checkout to origin's current branch.
. "$(dirname "$0")/common.sh"
git pull --ff-only
git log -1 --format='Now at %h %s (%cr)'
