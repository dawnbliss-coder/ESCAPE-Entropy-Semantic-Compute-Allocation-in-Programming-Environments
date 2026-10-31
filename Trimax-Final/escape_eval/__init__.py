"""Stream C evaluation library: the minimum PRELIMINARY slice for the mid submission.

Modules:
  data         file tables (corpus or golden_fixtures layout) and loaders
  targets      unique start/end targets with outermost-node attribution
  matching     vectorised tolerance-match counts for all strata at once; span IoU
  baselines    density-matched null H0 sampler, whitespace baseline
  stats        pooled metrics, permutation p-values, cluster bootstrap, paired randomisation
  engine       streaming per-file engine accumulating resample vectors per stratum
  calibrate_k  CLI: tolerance k per language on the calib split only
  run_eval     CLI: P1, P2, P3, P4, H0, whitespace and IoU tables for one run
  figures      CLI: figures drawn from results/ only
  results_io   provenance (git, params) and atomic result-table writers

Every number produced here is PRELIMINARY. Definitions, rationale and open
decisions are in docs/STREAM_C_METHODS.md.
"""

PRELIMINARY = True
