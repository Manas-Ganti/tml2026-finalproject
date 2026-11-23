#!/usr/bin/env bash
# Message board for the team's agents. Protocol: docs/BOARD.md
# Messages live on the `board` branch, checked out in a gitignored worktree at .board/.
# Each sender appends only to board/<name>.md, so pushes never conflict.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
WT="$ROOT/.board"
BRANCH="board"
REMOTE="origin"

die() { echo "board: $*" >&2; exit 1; }

ensure_wt() {
  if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
    git -C "$ROOT" fetch -q "$REMOTE" "$BRANCH" 2>/dev/null || true
    if git -C "$ROOT" show-ref -q --verify "refs/remotes/$REMOTE/$BRANCH"; then
      git -C "$ROOT" worktree add -q -B "$BRANCH" "$WT" "$REMOTE/$BRANCH"
    elif git -C "$ROOT" show-ref -q --verify "refs/heads/$BRANCH"; then
      git -C "$ROOT" worktree add -q "$WT" "$BRANCH"
    else
      die "no '$BRANCH' branch locally or on $REMOTE. Ask the repo owner to create it."
    fi
  fi
}

sync() {
  # Fast-forward to the remote board. Rebase is safe: everyone only appends to their own file.
  git -C "$WT" pull -q --rebase "$REMOTE" "$BRANCH" 2>/dev/null || true
}

me() {
  [ -f "$WT/.me" ] || die "run 'scripts/board.sh init <name>' first (e.g. init manas-agent)"
  cat "$WT/.me"
}

next_id() {
  local name="$1" f="$WT/board/$1.md" n
  n=$(grep -c "^### $name-[0-9]\{3\} " "$f" 2>/dev/null || true)
  printf "%s-%03d" "$name" "$(( ${n:-0} + 1 ))"
}

append_and_push() {
  local name="$1" msg="$2" subject="$3" f="$WT/board/$1.md"
  mkdir -p "$WT/board"
  [ -f "$f" ] || printf "# Board: %s\n\nAppend-only. Only %s writes here. Protocol: docs/BOARD.md on main.\n" "$name" "$name" > "$f"
  printf "\n%s\n" "$msg" >> "$f"
  git -C "$WT" add "board/$name.md"
  git -C "$WT" commit -q -m "board($name): $subject"
  if ! git -C "$WT" push -q "$REMOTE" "$BRANCH" 2>/dev/null; then
    sync
    git -C "$WT" push -q "$REMOTE" "$BRANCH" || die "push failed; message committed locally in .board/, retry 'scripts/board.sh sync'"
  fi
}

# Print message blocks (### header to next ###) whose header matches an awk regex.
blocks() {
  local pattern="$1"; shift
  awk -v pat="$pattern" -v self="${BOARD_SELF:-}" '
    FNR == 1 { show = 0 }
    /^### / { show = ($0 ~ pat) && (self == "" || index($0, "### " self "-") != 1); if (show) print "" }
    show { print }
  ' "$@"
}

cmd="${1:-inbox}"; shift || true
case "$cmd" in
  init)
    [ $# -eq 1 ] || die "usage: init <name>   (e.g. manas-agent)"
    ensure_wt; sync
    echo "$1" > "$WT/.me"
    echo "board: you are '$1'. Your file: board/$1.md on branch '$BRANCH'."
    ;;
  sync)
    ensure_wt; sync
    git -C "$WT" push -q "$REMOTE" "$BRANCH" 2>/dev/null || true
    echo "board: synced"
    ;;
  inbox)
    ensure_wt; sync; name="$(me)"
    role_re=""
    if [ -f "$ROOT/docs/PLAN.md" ]; then
      # Roles whose row in PLAN.md §3 names this agent, e.g. "| **B — ..." ... "manas-agent"
      for r in A B C; do
        if grep -E "^\| \*\*$r " "$ROOT/docs/PLAN.md" | grep -q "$name"; then role_re="$role_re|$r"; fi
      done
    fi
    shopt -s nullglob; files=("$WT"/board/*.md)
    [ ${#files[@]} -gt 0 ] || { echo "board: empty"; exit 0; }
    BOARD_SELF="$name" blocks "→ (all|$name${role_re}) ·" "${files[@]}" | awk '
      /^### / { hdr=$0; getline st; if (st ~ /open/) { print hdr; print st; keep=1 } else keep=0; next }
      keep { print }'
    ;;
  read)
    ensure_wt; sync
    if [ $# -ge 1 ]; then cat "$WT/board/$1.md"; else cat "$WT"/board/*.md; fi
    ;;
  post)
    [ $# -eq 3 ] || die "usage: post <to> \"<subject>\" \"<body>\""
    ensure_wt; sync; name="$(me)"; id="$(next_id "$name")"
    msg="### $id · $(date '+%Y-%m-%d %H:%M') · → $1 · $2
**status:** open
$3"
    append_and_push "$name" "$msg" "$id $2"
    echo "board: posted $id"
    ;;
  reply)
    [ $# -eq 2 ] || die "usage: reply <msg-id> \"<body>\""
    ensure_wt; sync; name="$(me)"; id="$(next_id "$name")"
    sender="${1%-*}"
    msg="### $id · $(date '+%Y-%m-%d %H:%M') · → $sender · re: $1
**status:** open
$2"
    append_and_push "$name" "$msg" "$id re $1"
    echo "board: posted $id (reply to $1)"
    ;;
  close)
    [ $# -eq 1 ] || die "usage: close <msg-id>  (only your own messages)"
    ensure_wt; sync; name="$(me)"; f="$WT/board/$name.md"
    case "$1" in "$name"-*) ;; *) die "you can only close your own messages ($name-NNN)";; esac
    grep -q "^### $1 " "$f" || die "no message $1 in board/$name.md"
    awk -v id="$1" '
      /^### / { inmsg = (index($0, "### " id " ") == 1) }
      inmsg && /^\*\*status:\*\* open/ { sub(/open/, "closed"); inmsg = 0 }
      { print }' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    git -C "$WT" add "board/$name.md"
    git -C "$WT" commit -q -m "board($name): close $1"
    git -C "$WT" push -q "$REMOTE" "$BRANCH" || { sync; git -C "$WT" push -q "$REMOTE" "$BRANCH"; }
    echo "board: closed $1"
    ;;
  *)
    die "unknown command '$cmd'. See docs/BOARD.md"
    ;;
esac
