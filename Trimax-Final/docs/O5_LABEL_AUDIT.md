# O5 label audit

The mid-report deferred the O5 inference "pending a label audit". This note is that audit. It checks the memory-unsafe region labels (`*`, `&`, `new`, `delete`, plus the unsafe-call and unsafe-cast labels) behind the frozen O5 result against tree-sitter, and records what the audit does and does not establish. The scorer and the frozen O5 numbers were not changed.

- **Inputs:** `results/final/c-v1-2721954/o5/o5_regions.csv` (frozen; 12,609 analysis regions in 1,500 C++ files), the restored corpus bytes, and `stream_b/build_o5_regions.py`. The audit never reads BLT outputs.
- **Code:** `scripts/ada_distill/o5_label_audit.py`, run on Ada by `scripts/ada_distill/o5_audit.sbatch` (job 43919, 2026-10-28; log `results/final/logs/o5_audit-43919.log`).
- **Outputs:** `results/final/o5_label_audit/summary.json` and `units.csv` (one row per audited region, with its source text).

## 1. Reproduction: all 1,500 files

The tree-sitter scan behind `build_o5_regions.py` was re-run on the restored bytes of every O5 file. Each frozen region was then checked for the same span, the same statement type, and the same `is_memory_unsafe` flag.

| Check | Result |
|---|---|
| Frozen regions found again with the identical byte span | 12,609 / 12,609 |
| `is_memory_unsafe` flag identical | 12,609 / 12,609 |
| Statement type (`declaration` / `expression_statement` / `return_statement`) identical | 12,609 / 12,609 |
| Files reproduced exactly | 1,500 / 1,500 |

The frozen region table therefore matches the committed labeller exactly on the restored corpus, with no drift since the Ada run.

## 2. Validity: 50 files, compared against an independent labeller

Reproduction only shows that the labeller is deterministic, not that its labels are right. For validity, 50 O5 files were drawn by seeded SHA256 rank (`SHA256(20260927, "o5-audit", file_id)`). Every frozen region in those files (435 units, 185 of them unsafe) was then labelled twice:

- **tree-sitter:** the operator of each `pointer_expression` (`*` or `&`), plus `new_expression`, `delete_expression`, unsafe calls and unsafe casts, exactly as `build_o5_regions.py` uses them.
- **lexical:** a token-level labeller that shares no code with tree-sitter. It treats `*` and `&` as unary when the previous token cannot end an operand, recognises `new` / `delete` not preceded by `operator`, unsafe calls by name plus `(`, and C-style pointer casts by the shape `( type-words *+ ) operand`.

| Label | Both | tree-sitter only | lexical only | Neither |
|---|---:|---:|---:|---:|
| `*` (dereference) | 62 | 0 | 0 | 373 |
| `&` (address-of) | 72 | 2 | 0 | 361 |
| `new` | 26 | 0 | 0 | 409 |
| `delete` | 11 | 0 | 0 | 424 |
| unsafe call (`malloc`, `memcpy`, `strcpy`, ...) | 12 | 0 | 0 | 423 |
| unsafe cast (`reinterpret_cast`, `const_cast`, C-style pointer cast) | 17 | 0 | 0 | 418 |

The two labellers agree on every label of 433 of the 435 units (99.5%). Both disagreements come from the same file and the same construct:

| Unit | tree-sitter | lexical | Source text |
|---|---|---|---|
| `cpp_e20509df1736:990` | `&` `cast` | `cast` | `return jhash2((uint32_t*)&s, sizeof(s) / sizeof(uint32_t), 0xbeefdead);` |
| `cpp_e20509df1736:1494` | `&` `cast` | `cast` | `ret = getnameinfo((const struct sockaddr *)&sa, sizeof(sa), hostname, 48, NULL, 0, NI_NAMEREQD);` |

**Adjudication:** tree-sitter is right in both cases. `&s` and `&sa` are address-of operations whose result is then cast. The lexical labeller misses them because the token before `&` is the `)` that closes the cast, and its rule treats `)` as the end of an operand. These are lexical-labeller errors, not O5 label errors. Neither unit changes its `is_memory_unsafe` flag, since both are unsafe through the cast anyway.

No comparison (safe) unit carried an unsafe construct under either labeller: every "lexical only" cell is 0. The audit therefore found no false negatives in the safe group of this sample.

## 3. What the audit establishes, and what it does not

- The region spans, statement types and unsafe flags are correct as **syntax**: within these 50 files the unary `*`, `&`, `new` and `delete` labels match an independent reading of the source exactly. No relabelling is needed, and the frozen O5 regression stands as computed.
- The labels are syntactic by design (`build_o5_regions.py`: "This labels syntax, not program safety"). No types, aliasing or bounds are known, so "memory-unsafe" means "contains a raw-pointer, manual-allocation or unchecked-copy construct", not "contains a memory bug".
- O5 uses only C++ files whose tree-sitter parse needed no error recovery (`parse_ok`). Files with macro-heavy, error-recovered parses, where labels would be least reliable, are therefore excluded upstream rather than audited here.
- **Interpretation of the O5 result:** with labels confirmed, the question is how much of the effect survives the identifier confound. `scripts/final_analysis/o5_confound.py` answers this (output `results/final/analysis/o5_confound.json`). Without the identifier controls, unsafe regions carry *fewer* boundaries than comparison regions in the same file (−0.0083 per byte, 95% CI [−0.0127, −0.0039]). The frozen +0.0069 [0.0043, 0.0095] appears only after conditioning on mean identifier entropy, because unsafe regions use more predictable identifiers (0.49 vs 0.55 nats). The conditional effect is 4.2% of the comparison regions' mean boundary density, and 7% of the effect of one standard deviation of identifier entropy. `docs/FULL_RUN_RESULTS.md` §O5 gives the reading.
