# Team working agreements

## Structure and data

- `src/scripts/`: shared training, evaluation, detection, and tracking code.
- `src/models/`: only selected team models. Name trained weights with date, trainer, and descriptor; keep the upstream base model name.
- `src/runs/`: generated outputs; do not commit. Keep historical `runs/` ignored too.
- `data/`: local datasets and footage; do not commit or send private Primagaz data to external services.

## Workflow

- Track every task, including non-coding work, with a GitHub Issue.
- Use a separate branch named `<issue-number>-<name>-<description>`.
- Open a focused PR linked to the Issue. Require one teammate approval and resolved conversations before merging into `main`.
- Preserve teammates' work and existing local data. Do not force-push shared branches or rewrite shared history.
- Keep the README's layout, commands, Models section, and workflow current.

## Code and AI assistance

- Keep code simple, concise, portable, and understandable. Avoid unnecessary options, abstractions, and duplicated settings.
- Follow course AI-use rules and preserve learning and authorship. Explain meaningful changes and verify behavior.
- Implement only explicitly requested changes. Commit, push, and merge only when explicitly authorized.
- Review code simplicity with the developer before any push, and review the final PR diff again before merging.
