#!/usr/bin/env bash
# push-results [job]: copy small result files from runs/<job>/ (all jobs if none given) to
# results/gaming-pc/<job>/ on the branch gaming-pc-results, commit and push it. The steward merges that
# branch. Only listed file types under 1 MB; checkpoints and anything else stay on this machine.
. "$(dirname "$0")/common.sh"
BRANCH=gaming-pc-results
WT="$REPO/runs/_gaming-pc/results-worktree"   # a separate checkout, so this one stays on main for sync
jobs="${1:-$(cd runs && ls -d */ 2>/dev/null | sed 's#/$##' | grep -v '^_' || true)}"
[ -n "$jobs" ] || die "No results in runs/."
for j in $jobs; do [[ "$j" =~ ^[A-Za-z0-9_-]+$ && -d "runs/$j" ]] || die "No results folder runs/$j."; done

git fetch -q origin
if [ ! -d "$WT" ]; then
  if git rev-parse -q --verify "origin/$BRANCH" > /dev/null; then git worktree add -q -B "$BRANCH" "$WT" "origin/$BRANCH"
  else git worktree add -q -B "$BRANCH" "$WT" origin/main; fi
fi
git -C "$WT" pull -q --ff-only origin "$BRANCH" 2>/dev/null || true

for j in $jobs; do
  (cd "runs/$j" && find . -path '*/checkpoint*' -prune -o -type f -size -1048576c \
     \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.txt' -o -name '*.png' -o -name 'DONE' \) -print) |
  while read -r f; do
    mkdir -p "$WT/results/gaming-pc/$j/$(dirname "$f")"
    cp "runs/$j/$f" "$WT/results/gaming-pc/$j/$f"
  done
done
cd "$WT"
git add results/gaming-pc
if git diff --cached --quiet; then echo "Nothing new to push."; exit 0; fi
git diff --cached --stat | tail -n 20
git commit -q -m "Results from gaming-pc: $(echo $jobs)"
git push -q -u origin "$BRANCH"
echo "Pushed $(git log -1 --format=%h) to $BRANCH."
