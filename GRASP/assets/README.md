# Partitions and splits

`groups/<dataset>.npy` contains an integer group ID for every node, in dataset
node order. The eight smaller datasets include the final CHROMA partition.
`groups/ogbn-arxiv_initial.npy` contains the initial independent-set partition;
run `prepare_groups.py --dataset ogbn-arxiv` to complete merging and selection.

`splits/` contains the fixed Amazon and Coauthor split indices. Each `.npz`
file has `train`, `valid`, and `test` arrays. The runner checks for conflicting
split files before copying them into the data directory.

`manifest.json` records SHA-256 checksums for these inputs, the experiment
configurations, and the per-seed reference results:

```bash
python scripts/verify_data.py
```
