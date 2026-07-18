# Asset policy

`migrated-assets.json` is generated from the actual handover tree and records critical path sizes, file counts, and hashes. `external-assets.toml` lists intentionally omitted heavyweight dependencies and their expected landing paths.

The repository must never resolve a path through the former `Projects` checkout. Base model downloads and Docker image restore are explicit provisioning steps; training outputs remain outside version control.

The handover directory is one Git repository. Its local commit tracks code,
documentation, configuration, and lightweight manifests; the migrated 8+ GB
asset bundle is intentionally ignored rather than written into Git objects.
Decide on Git LFS versus independent artifact storage before publishing;
distribute the whole checkout in the meantime and use `migrated-assets.json`
to verify the ignored assets.
