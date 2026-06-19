# Security policy

V-GIB is research software for representation-learning experiments. Do not upload private datasets, credentials, tokens, or sensitive data to issues, pull requests, or example runs.

## Reporting security concerns

Open a GitHub issue only for non-sensitive bugs. For sensitive concerns, contact the repository maintainer privately.

## Data handling

The repository should not commit generated datasets, model checkpoints, private logs, or downloaded benchmark data. Generated artifacts should stay under ignored folders such as `data/`, `runs/`, `logs/`, `figures/`, and `tables/`.
