# Asset policy

`migrated-assets.json` is generated from the actual handover tree and records critical path sizes, file counts, and hashes. `external-assets.toml` lists intentionally omitted heavyweight dependencies and their expected landing paths.

The repository must never resolve a path through the former `Projects` checkout. Base model downloads and Docker image restore are explicit provisioning steps; training outputs remain outside version control.

The repository tracks code,
documentation, configuration, and lightweight manifests; the migrated 8+ GB
asset bundle is intentionally ignored rather than written into Git objects.
The bundle is published as the Hugging Face dataset
[`simonlqy/SkillGate-Assets`](https://huggingface.co/datasets/simonlqy/SkillGate-Assets)
with repository-mirrored paths — restore it from the repo root with
`hf download simonlqy/SkillGate-Assets --repo-type dataset --local-dir .` and
verify with `migrated-assets.json`.
