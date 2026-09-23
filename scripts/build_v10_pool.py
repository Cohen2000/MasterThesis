#!/usr/bin/env python3
"""Generate the fixed synthetic training and development pool for v10."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import REFERENCES, write_json
from main_experiment.pool import build_pool, pool_definition

definition = pool_definition()
REFERENCES.mkdir(parents=True, exist_ok=True)
write_json(REFERENCES / 'pool_definition.json', definition)
build_pool(REFERENCES / 'pool', definition['graphs'])
print('V10_POOL', definition['n_graphs'])
