# Data Contract (DRAFT — review with A & C before freezing)

This is what lets the three streams run in parallel: agree it once, freeze it, and every
downstream file traces back to a `file_id` and a byte offset defined here. Column names
below are a starting point — the discipline (byte offsets, never re-encoded text, one
`file_id` per source file) matters more than the exact names.

## Conventions

- **Offsets are byte offsets into the frozen UTF-8 bytes in `corpus/`, never character
  offsets.** Any tool that only produces character offsets (e.g. a prose constituency
  parser) must round-trip them to byte offsets and that conversion must be validated
  against the golden fixtures before its output is trusted.
- `domain` ∈ `{py, cpp, prose}`.
- `split` ∈ `{calib, main}` — the calibration split is used to fix tolerance `k` a priori,
  per language, before any alignment number is generated.

## B → A, C — the canonical text

```
corpus/{domain}/{file_id}.bin      raw UTF-8 bytes, never re-encoded downstream
corpus/manifest.parquet            file_id, domain, n_bytes, sha256, split
```

## B → C — structure

```
structure/{domain}/{file_id}.parquet   file_id, node_type, parent_type, depth, start_byte, end_byte
whitespace/{domain}/{file_id}.parquet  file_id, byte_offset            (newline + indent-change positions)
memory_regions/cpp/{file_id}.parquet   file_id, kind ∈ {deref, addr_of, new, delete}, start_byte, end_byte
identifiers/{domain}/{file_id}.parquet file_id, start_byte, end_byte
taxonomy.json                          node_type -> {deterministic_opener | open_ended}
```

## B → A — the adversarial condition (not needed until ~Week 7)

```
adversarial/{domain}/{file_id}.bin     mutated bytes
adversarial/manifest.parquet           file_id, clean_file_id, mutation_kind, n_mutations
```

## A → C — boundaries and entropy

```
entropies/{run_id}/{file_id}.npy       float32[n_bytes], per-byte entropy in nats
boundaries/{run_id}/{file_id}.parquet  file_id, byte_offset, trigger ∈ {entropy, length_cap, init}, entropy
bpp_summary.parquet                    file_id, run_id, n_bytes, n_patches, mean_bpp, var_bpp
runs.json                              run_id -> {tau, sliding_window, max_patch_length, checkpoint, commit}
```

## C → everyone — the shared library

```
escape_common/    schema definitions, offset utilities, file_id conventions, validator
escape_eval/      scorer, baselines, permutation engine, figures
results/          output tables for the report
```

## Golden fixtures

`golden_fixtures/` holds ~20 hand-verified files (offsets checked by eye) covering both
languages. This is the regression test all three streams run against their own code —
if your output disagrees with the golden fixtures, your code is wrong, not the fixtures.
