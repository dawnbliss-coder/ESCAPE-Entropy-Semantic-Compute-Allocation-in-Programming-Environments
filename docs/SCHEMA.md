# Stream B Data Contract

What this is: the on-disk schema that lets Stream A (BLT patcher), Stream B
(corpus/structure) and Stream C (scoring) run independently and still agree on
every file. Frozen early in the project and mirrored in code by
`escape_common/schema.py`; the discipline it fixes (byte offsets, never
re-encoded text, one `file_id` per source file) is still exactly what the
pipeline implements. The O4/O5 sections below were added later, once those
pipelines were built, to keep this document matching the real output paths.

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
corpus/manifest.parquet            file_id, domain, n_bytes, sha256, split, parse_ok
                                    (parse_ok: false if tree-sitter's error-recovery
                                    triggered anywhere in the file — kept, not excluded;
                                    see stream_b/build_structure.py for why)
```

## B → C — structure

```
structure/{domain}/{file_id}.parquet   file_id, node_type, parent_type, depth, start_byte, end_byte
                                        (depth: full-tree depth from root=0, every
                                        ancestor counted — not just tracked node types.
                                        For domain=prose specifically: node_type is a
                                        benepar constituent label, or several joined with
                                        '+' when NLTK/benepar collapses a unary chain
                                        (e.g. "S+VP"); depth=1 is each sentence's root
                                        constituent (a paragraph has no single file-level
                                        root the way code has module/translation_unit, so
                                        depth=1 keeps it comparable to code's top-level
                                        statements); parent_type="paragraph" is a synthetic
                                        placeholder for a sentence root, not a real benepar
                                        node. Raw depth is not perfectly comparable
                                        cross-domain because of the unary-collapse — see
                                        stream_b/build_prose_structure.py.)
whitespace/{domain}/{file_id}.parquet  file_id, byte_offset, kind ∈ {newline, indent_change}
                                        - newline: position right after every `\n`
                                          (the required P1 whitespace baseline — does
                                          NOT skip leading indentation, see proposal §5.3)
                                        - indent_change: position of the first real
                                          content on lines whose indentation depth
                                          differs from the previous non-blank line
                                          (feeds the R2 quantification, additional to
                                          the newline baseline, not a replacement)
taxonomy.json                          node_type -> {deterministic_opener | open_ended}
```

## B → C — O5 inputs (C++ only; `stream_b/build_o5_regions.py`)

```
analysis_regions/cpp/{file_id}.parquet  file_id, start_byte, end_byte, statement_type
memory_regions/cpp/{file_id}.parquet    file_id, start_byte, end_byte, unsafe_kind
identifiers/cpp/{file_id}.parquet       file_id, start_byte, end_byte
analysis_regions/cpp/_selection.json    the sampling rule, seed and unit counts
```

## B → A — O4 noise conditions (`stream_b/build_o4_perturbations.py`)

```
o4/manifest.parquet                        file_id, condition, ... (per-condition edit record)
o4/{condition}/text/{sample_id}.bin        noised bytes for one condition
o4/{condition}/structure/{sample_id}.parquet  clean-AST targets remapped onto the noised bytes
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
escape_eval/      earlier, exploratory scorer: baselines, permutation engine, figures
escape_scoring/   frozen C-v1 scorer used for the submitted results (docs/SCORING_PROTOCOL.md)
results/          output tables for the report
```

## Golden fixtures

`golden_fixtures/{domain}/{file_id}.bin` + `{file_id}.structure.parquet` (+
`{file_id}.whitespace.parquet` for py/cpp) hold 18 hand-picked, hand-verified files
(6 py, 6 cpp, 6 prose — content read and spot-checked by eye before selection, not
just picked by size) with frozen expected output. This is the regression test all
three streams run against their own code: the frozen fixtures are treated as
ground truth, so a mismatch means the code regressed, not that the fixtures need
updating. `stream_b/validate_golden_fixtures.py` re-runs Stream B's own
extraction against these and fails loudly on any mismatch; A and C have an
equivalent check for whatever they consume from these files.
