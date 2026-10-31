"""Stream A: BLT patcher -> per-byte entropy -> patch boundaries -> BPP.

Consumes only raw corpus bytes; never reads structure/, whitespace/ or any parse
information (AST data is ground truth for Stream C, never an input here)."""
