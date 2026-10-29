# Python project guidelines

Use a virtual environment for each Python project. Pin major dependency ranges in requirements.txt and document the supported Python version. Separate application code, tests, data, and scripts when this improves maintainability.

For automated tests, use pytest. Unit tests should verify small units of behavior without depending on a running remote service. Keep test data minimal and deterministic. Do not commit credentials, local environment files, or large generated artifacts to source control.

When reporting model performance, record the dataset, split strategy, metric, model version, and the conditions under which the experiment ran. A metric without an evaluation setup is difficult to reproduce or compare.
