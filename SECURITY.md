# Security Policy

Report vulnerabilities with GitHub's private vulnerability reporting feature.
If that feature is unavailable, open a public issue asking for a private
contact channel without including vulnerability details.

- Never deserialize untrusted model or data files.
- Keep API input and output roots narrow.
- Process untrusted video in isolated, resource-limited workers.
- Validate media duration, resolution, size, and codec.
- Do not publish private footage, credentials, or proprietary weights.
- Rotate any exposed credential immediately and inspect repository history.
