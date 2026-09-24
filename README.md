## Build progress

Built step by step. Every component has a tag, and every commit is scoped to it:

- `S` setup · `I` infrastructure · `A` agents (pipeline order) · `G` human gates · `L` go-live · `M` milestones
- Commit format: `type(TAG): message`, e.g. `feat(A01): repo_ingest with SHA cache`
- Filter history by component: `git log --oneline --grep="A01"`# content-os
Multi-agent pipeline that turns GitHub repos into evidence-bound LinkedIn posts
