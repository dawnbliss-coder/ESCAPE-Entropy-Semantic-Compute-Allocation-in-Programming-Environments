# Same as `python -m stream_b.build_ext_corpus ARGS`, but skips interpreter finalization:
# job 39365 aborted there (PyGILState_Release, exit 134) after a complete write of all
# .bin files, manifest_ext.parquet and DATA_NOTE.json.
import os
import sys

from stream_b.build_ext_corpus import main

rc = main(sys.argv[1:])
sys.stdout.flush()
sys.stderr.flush()
os._exit(rc or 0)
