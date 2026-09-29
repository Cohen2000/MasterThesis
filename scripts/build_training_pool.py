#!/usr/bin/env python3
"""Generate the fixed synthetic training and development pool for ExtraTrees."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import REFERENCES, write_json
from main_experiment.pool import build_pool, pool_definition

# The pool = a fixed set of synthetic graphs (with known true persistence) that
# ExtraTrees learns from, in addition to the real training sources. The definition
# (generator settings and seeds) is saved first, then every pool graph is generated.
definition = pool_definition()
REFERENCES.mkdir(parents=True, exist_ok=True)
write_json(REFERENCES / 'pool_definition.json', definition)
build_pool(REFERENCES / 'pool', definition['graphs'])
print('POOL', definition['n_graphs'])
