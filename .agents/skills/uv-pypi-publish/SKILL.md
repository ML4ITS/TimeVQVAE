---
name: uv-pypi-publish
description: Publish a Python package to PyPI using uv with credentials loaded from a local `.secrets` file. Use when asked to release a package after building, to run a clean rebuild (`dist/` reset), and to publish with `UV_PUBLISH_TOKEN` from environment variables.
---

# Uv Pypi Publish

Execute this workflow in order when publishing:

1. Load secrets:
```bash
source .secrets
```

2. Remove old build artifacts:
```bash
rm -rf dist/
```

3. Build package:
```bash
uv build
```

4. Publish with token:
```bash
UV_PUBLISH_TOKEN=$UV_PUBLISH_TOKEN uv publish
```

If `UV_PUBLISH_TOKEN` is empty after sourcing, stop and report that publishing cannot proceed without a valid token.
