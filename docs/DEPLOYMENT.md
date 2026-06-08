# Deployment Guide

## Local

```bash
pip install -e .
matchtrace demo
matchtrace serve
```

## Production

1. Build an immutable Python 3.11 image.
2. Install exact dependency versions from a validated lock file.
3. Mount approved detector and classifier artifacts read-only.
4. Mount separate input, output, and log volumes.
5. Run the API as an unprivileged user.
6. Put long video jobs behind a queue and worker timeout.
7. Restrict upload size, MIME type, duration, and resolution.
8. Monitor latency, failures, disk usage, and model version.

## Environment

`MATCHTRACE_MODEL_PATH` optionally overrides the real-video detector path.
No database or external service is required for the local workflow.

## Verification

- Call `GET /health`.
- Run a known synthetic fixture.
- Compare frame count, accuracy, and summary schema.
- Confirm output and logs remain within configured roots.
- Verify model and dependency licenses.

## Rollback

Keep the previous image, TOML configuration, and model checksum. Stop accepting
new jobs, drain active workers, restore the previous version, run the synthetic
fixture, and resume traffic only after health and output checks pass.

