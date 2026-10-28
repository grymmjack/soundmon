---
name: create-release
description: Cut a soundmon release — bump the version, update docs, verify (selfcheck, a smoke render on the rtx box with an fp32 check and a spectral check, the GUI), then commit + tag + push and publish the GitHub Release with curated notes
---

# Release Skill (soundmon)

When the user invokes this skill (e.g. "do a release", "create release", "release skill", "push and create release"), execute the steps **in order**. Do not skip steps. Do not batch steps that need user input.

soundmon is a **Python CLI** (`soundmon.py` + the CPU engines `chip.py`, `opl.py`, `chipfx.py`, `blip.py`, `narrate.py`, `record.py`, `master.py`, `similar.py`) + a **stdlib web GUI** (`gui/server.py` + a single-page `gui/index.html`) + **a ComfyUI custom node** (`custom_nodes/retro_sfx/`). There is **no CI**: nothing builds or publishes on a tag push. A release is **a `vX.Y.Z` git tag plus a GitHub Release created with `gh release create`** (GitHub attaches the source archives automatically). The Release notes are exactly what Step 7 writes.

Repo: `grymmjack/soundmon`. Default branch: `main`. Sibling: `grymmjack/pixelmon` (same ComfyUI, same farm).

> **You cannot hear the output.** Every check below is a proxy (AGENTS.md, "Every one of those checks can pass on unusable audio"). Say what was measured, and that it has not been heard. If the release touched an AI engine, ask the user to listen to the smoke render before tagging.

---

## Step 1 — Determine the version bump

The version lives in `soundmon.py`:

```python
__version__ = "X.Y.Z"
```

`soundmon --version` prints it. The first tagged release is **0.1.0** (the version that introduced `--version`); if there is no `v*` tag yet, propose `0.1.0` and skip the question.

Otherwise ask the user (unless they already said which):

> "Is this a **major**, **minor**, or **patch** release?"
> (major = breaking, minor = new features, patch = fixes / small features)

- **major** → `(X+1).0.0`  ·  **minor** → `X.(Y+1).0`  ·  **patch** → `X.Y.(Z+1)`

The project is pre-1.0: suggest **minor** for new features (a new engine, new flags, new GUI tab, new node options) and **patch** for fixes. Breaking = a removed/renamed CLI flag, a changed `key | text` manifest or `set.json` format that old files can't load, a renamed `RetroSFX`/`SaveSFX` input (saved ComfyUI workflows and every farm box break), or a changed `✅ [i/n] … seed=` result line (the GUI parses it).

Show `Current: X.Y.Z → New: A.B.C` and confirm before proceeding.

---

## Step 2 — Collect changes

```bash
cd <the soundmon checkout>
LAST=$(git describe --tags --abbrev=0 2>/dev/null)
git log --oneline --no-merges ${LAST:+$LAST..}HEAD
```

(No tag yet → this lists the whole history; summarise it as the first release.)

Merged PRs since the last tag, if any:

```bash
gh pr list --repo grymmjack/soundmon --state merged --limit 20 \
  --json number,title,mergedAt --jq '.[] | "PR #\(.number): \(.title)  \(.mergedAt[0:10])"'
```

Also review the **current conversation** for work done this session that isn't committed yet.

Produce a deduplicated, categorised list: **New Features · Improvements · Bug Fixes · Internal · Breaking Changes**. Keep it **user-facing**: name CLI flags, engines and GUI tabs the way a user sees them.

---

## Step 3 — Bump the version

Edit `__version__` in `soundmon.py` to `A.B.C`. Confirm:

```bash
python3 soundmon.py --version     # → soundmon A.B.C
```

---

## Step 4 — Update the docs

Review against the Step 2 list. Update only what's stale; don't rewrite accurate sections.

- `README.md`: feature list, the engine sections, flag tables, the GUI section, Repo layout.
- `AGENTS.md`: any new gotcha learned this cycle; the architecture notes if a stage was added.
- `print_help()` in `soundmon.py`: **every flag must appear** — `tools/selfcheck.py` fails otherwise.
- `download-models.sh` header and `--help`: if a model or group changed.

---

## Step 5 — Verify

**Before touching the rtx box, check nobody is rendering** (the user's GUI queue, and rtx's ComfyUI queue):

```bash
curl -s 127.0.0.1:8191/api/jobs | python3 -c "import json,sys; print([j['status'] for j in json.load(sys.stdin)['jobs'] if j['status'] in ('queued','running')])"
curl -s http://192.168.1.77:8188/queue | python3 -c "import json,sys; q=json.load(sys.stdin); print(len(q['queue_running']), len(q['queue_pending']))"
```

Smoke renders queue fine behind other jobs, but **never restart rtx's ComfyUI while anything is queued or running** — pixelmon may be using it.

1. **Syntax and the offline invariants**:
   ```bash
   for f in *.py gui/*.py tools/*.py custom_nodes/retro_sfx/*.py; do python3 -m py_compile "$f" || echo "FAIL $f"; done
   python3 -c "import re; s=open('gui/index.html').read(); open('/tmp/sm-gui.js','w').write(re.search(r'<script>(.*)</script>', s, re.S).group(1))" && node --check /tmp/sm-gui.js
   bash -n bin/soundmon bin/soundmon-gui install.sh download-models.sh launch-comfyui.sh
   python3 tools/selfcheck.py            # must be all-pass
   python3 soundmon.py --help >/dev/null
   ```
2. **The server's dtype** — Stable Audio 3 renders static in fp16, quickly and without an error (AGENTS gotcha 11). rtx's ComfyUI must have been started with `--force-fp32`:
   ```bash
   curl -s http://192.168.1.77:8188/system_stats | python3 -c "import json,sys; a=json.load(sys.stdin)['system'].get('argv'); print(a); print('OK' if a and '--force-fp32' in a else 'NOT fp32 — restart it with --force-fp32 (Step 5.4) before the smoke render')"
   ```
   soundmon itself prints a `⚠ … started WITHOUT --force-fp32` line in that case; a smoke render without it proves nothing.
3. **Smoke render on rtx** (**always `--no-open`**: without it soundmon plays the sound on the user's machine):
   ```bash
   python3 soundmon.py "a heavy wooden door creaking open" --server rtx --no-open --seed 7 --seconds 4 \
     --output-to /tmp/sm-smoke --create-dirs
   python3 tools/spectral-check.py /tmp/sm-smoke/*.wav --min 20     # full-band SA3 reads ~20 kHz; a missing top end is the "processed, frequencies gone" fault
   ffprobe -v error -show_entries format_tags=comment -of default=nw=1:nk=1 /tmp/sm-smoke/*.wav   # the recipe is embedded
   ```
   Check the `✅ [1/1] seed=7 -> …` line, the duration, and that the recipe JSON shows the prompt, seed and command. If this release touched a path (`--song`, `--music`, `--ref`, `--loop`, `--batch-file`, a farm change), smoke that path too — `--song` needs the song model on the box, and `--server rtx,<other>` exercises the farm.
   Then **ask the user to listen** to the WAV before tagging.
4. **Custom node changes** (`custom_nodes/retro_sfx/`): every farm box loads the node from its own checkout, and a box with a stale node rejects graphs that use a new input. Copy the changed node files to each box (don't `git pull` over local edits there), then — only when both queues above are empty — restart its ComfyUI **with `--force-fp32`**:
   ```bash
   ssh daw 'tmux kill-session -t comfy; sleep 2; tmux new-session -d -s comfy "bash -lc \"~/launch-comfyui.sh --force-fp32\""'
   ```
   Then confirm `curl -s http://192.168.1.77:8188/object_info/RetroSFX` answers, and re-run the smoke render.
5. **CPU engines** (no server): one each, so a hand-off regression shows:
   ```bash
   for e in --chip --opl --chipfx --blip; do python3 soundmon.py "dungeon door" $e --seconds 4 --no-open --output-to /tmp/sm-smoke-cpu --create-dirs | tail -1; done
   ```
   (`--opl` needs `./download-models.sh --opl` on this machine.)
6. **GUI**: start a throwaway server on a spare port, so the user's running `soundmon-gui` on 8191 is untouched:
   ```bash
   (timeout 30 python3 gui/server.py --port 8199 >/dev/null 2>&1 &); sleep 3
   curl -s -o /dev/null -w "%{http_code}\n" 127.0.0.1:8199/api/meta          # 200
   ```
   If the UI changed, screenshot the affected tab headlessly:
   ```bash
   google-chrome --headless=new --disable-gpu --no-sandbox --window-size=1500,1200 \
     --virtual-time-budget=6000 --screenshot=/tmp/sm-gui.png "http://127.0.0.1:8199/"
   ```

Fix any real failure before continuing.

---

## Step 6 — Commit, tag, push

**Confirm with the user before this step.** The tag and the Release are public.

Never `git add -A` blindly: `servers.json` (real LAN IPs), `vendor/` (LGPL/GPL sources fetched at setup) and personal files sit in the tree. Stage the release's files explicitly and check the diff:

```bash
git status --short
git diff --stat                      # only the intended files
git add soundmon.py README.md AGENTS.md …   # explicit paths
git commit -m "Release vA.B.C"       # end with the Co-Authored-By / Claude-Session trailers from the system reminder
git push origin main
git tag -a vA.B.C -m "soundmon vA.B.C"
git push origin vA.B.C
```

---

## Step 7 — Publish the GitHub Release

Nothing publishes it automatically; create it with the curated Step 2 notes:

```bash
gh release create vA.B.C --repo grymmjack/soundmon --title "soundmon vA.B.C" --notes-file - <<'EOF'
## soundmon vA.B.C — YYYY-MM-DD

### New Features
- …

### Improvements
- …

### Bug Fixes
- …

### Breaking Changes
- …
EOF
```

Drop empty sections. Print the same markdown in chat, and the Release URL (`gh release view vA.B.C --repo grymmjack/soundmon --json url --jq .url`).

---

## Rules

- **Always confirm** the version bump (Step 1) and the tag push + Release (Steps 6–7). Never assume either.
- The tag and the Release are public, outward-facing actions. Treat them that way.
- **Always pass `--no-open`** to any soundmon render.
- **Never restart a ComfyUI while a render is queued or running** (check both queues in Step 5), and always restart it **with `--force-fp32`**.
- The rtx box is the render box to use (`--server rtx`); don't render on the local GPU.
- Never report audio as good because the checks passed: say what was measured, and get it heard.
- Keep the diff scoped: stage explicit paths, never `git add -A`.
- If a step produces no changes (README already accurate, no node changes), say so and move on.
- Release-notes date = today, `YYYY-MM-DD`.
