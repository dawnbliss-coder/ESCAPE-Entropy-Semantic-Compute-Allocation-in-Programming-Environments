"""Print the file_ids of one extraction shard: files whose corpus bytes are present,
split by longest-processing-time assignment on n_bytes, so shards finish together.
Deterministic: depends only on the manifest and which .bin files exist."""
import sys
from pathlib import Path

import pandas as pd

shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
m = pd.read_parquet("corpus/manifest.parquet")
m = m[[Path(f"corpus/{d}/{f}.bin").exists() for d, f in zip(m["domain"], m["file_id"])]]
load = [0] * n_shards
mine = []
for fid, n in sorted(zip(m["file_id"], m["n_bytes"]), key=lambda x: (-x[1], x[0])):
    target = min(range(n_shards), key=lambda s: (load[s], s))
    load[target] += n
    if target == shard:
        mine.append(fid)
print(" ".join(sorted(mine)))
