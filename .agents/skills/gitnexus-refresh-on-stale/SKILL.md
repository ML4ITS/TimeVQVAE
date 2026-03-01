---
name: gitnexus-refresh-on-stale
description: Refresh GitNexus indexing and regenerate local GitNexus skills when repository status is stale. Use when `npx gitnexus status` reports stale/outdated index state, or when you need to sync newly generated `.claude/skills/gitnexus` into `.agents/skills/gitnexus` with overwrite behavior.
---

# GitNexus Refresh On Stale

Run this skill from the repository root.

## Workflow

1. Check index status.
```bash
npx gitnexus status
```

2. If status contains `stale` (case-insensitive), run:
```bash
npx gitnexus analyze
```

3. If `.claude/skills/gitnexus` exists after analyze, replace `.agents/skills/gitnexus` with it.
- Delete existing `.agents/skills/gitnexus` when present.
- Move `.claude/skills/gitnexus` to `.agents/skills/gitnexus`.

4. If status is not stale, do not run analyze.

## Script

Use the bundled script for deterministic behavior:
```bash
bash .agents/skills/gitnexus-refresh-on-stale/scripts/refresh_gitnexus_skills.sh
```

The script performs all steps above and prints what it changed.
