# Contributing

Create focused changes with tests and documentation. Do not commit private
footage, credentials, proprietary model weights, or generated artifacts.

Before opening a pull request:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src app tests
```

Document model, dataset, threshold, schema, or API behavior changes. Preserve
the attribution record and include licenses for new third-party assets.

