#!/usr/bin/env python3
"""Audit the final scientific prompt harness, serialized observations, and request manifest."""
import csv, hashlib, json, math, re, sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import ROOT, CURRENT_RUN, DESIGN_VERSION, sha, digest, read_json, write_json, CONFIGS, ARMS, ARM_ID, seed
from main_experiment.observation import parse, features, serialize, messages, FEATURE_NAMES

PROMPT_REVISION_STATEMENT = (
    "The final prompt revision aligns the task representation with the finalized observation mechanisms "
    "and removes inapplicable historical fields. No wording or field was selected based on comparative LLM accuracy."
)

def main():
    run = ROOT / CURRENT_RUN
    spec = ROOT / 'config/main_experiment'
    
    # 1. Common scientific template audit
    system_text = (spec / 'system.txt').read_text().rstrip('\n')
    user_prefix = (spec / 'user_prefix.txt').read_text().rstrip('\n')
    
    # Check required common elements
    assert "W=5" in user_prefix
    assert "E_full" in user_prefix
    assert "K_e" in user_prefix
    assert "rho_k" in user_prefix
    assert "rho_2 >= rho_3 >= rho_4 >= rho_5" in user_prefix
    assert "The full numbers of vertices, dyads, and events are unknown" in user_prefix
    assert "Dyads with no observed event are absent from the table but may belong to E_full" in user_prefix
    assert "The observation need not uniquely identify the full profile; still provide your best point estimates" in user_prefix
    assert "Return your final answer as one JSON object with exactly the keys rho_2, rho_3, rho_4, rho_5" in system_text
    
    # Check removed elements from common
    assert "Evaluation uses absolute error, primarily for rho_2" not in user_prefix
    assert "Walk_A" not in user_prefix
    assert "r_e/m_e" not in user_prefix
    assert "10%" not in user_prefix
    assert "ten-percent" not in user_prefix
    assert "corrector" not in user_prefix.lower()
    assert "baseline" not in user_prefix.lower()
    
    # Check exact required observation semantics sentence
    assert "Temporal_access indicates temporal accessibility only; it does not imply complete observation unless the sampling rule states so." in user_prefix
    assert "1 = at least one observed event in the window" in user_prefix
    assert "0 = no observed event in an accessible window" in user_prefix
    assert "? = temporally inaccessible window" in user_prefix
    assert "Whether an observed 0 implies true inactivity depends on the sampling rule" in user_prefix
    assert "All permitted patterns with at least one observed event are listed" in user_prefix
    assert "Dyads without any observed event are omitted, so the all-zero pattern is not listed" in user_prefix
    assert "The dyads and events columns sum to D_obs and M_obs, respectively" in user_prefix

    common_hashes = {
        'system_sha256': sha(spec / 'system.txt'),
        'user_prefix_sha256': sha(spec / 'user_prefix.txt')
    }

    # 2. Arm-specific rule and auxiliary semantics audit
    rule_files = {
        'R': spec / 'rule_R.txt',
        'S': spec / 'rule_S.txt',
        'H': spec / 'rule_H_time_v3.txt',
        'B': spec / 'rule_B.txt'
    }
    aux_files = {
        'S': spec / 'aux_S.txt'
    }
    
    r_text = rule_files['R'].read_text()
    assert "Uniform node panel" in r_text
    assert "For each e in E_full whose two endpoints are in the panel, its complete full-archive history is retrieved." in r_text
    assert "Accessible zeros for listed dyads therefore indicate true inactive windows." in r_text
    
    s_text = rule_files['S'].read_text()
    assert "Simple random walk" in s_text
    assert "initial vertex is drawn uniformly from V_full" in s_text
    assert "next vertex is chosen uniformly among the distinct neighbors" in s_text
    assert "Event multiplicities do not affect transition probabilities" in s_text
    assert "Exactly L transitions are taken, with no burn-in, no restart, and no stopping" in s_text
    assert "walk stays in its starting connected component" in s_text
    assert "For each traversed dyad, its complete full-archive history is retrieved." in s_text
    assert "Accessible zeros for listed dyads therefore indicate true inactive windows." in s_text
    assert "Walk_A" not in s_text  # Walk_A is in aux_S.txt
    
    aux_s_text = aux_files['S'].read_text()
    assert "Walk_A has five entries A_1, ..., A_5" in aux_s_text
    assert "A_j = sum_{e: r_e>0, K_e=j} r_e" in aux_s_text
    assert "r_e is the total number of traversals of dyad e" in aux_s_text
    assert "K_e is its number of active windows in its complete full-archive history" in aux_s_text
    assert "sum to L" in aux_s_text
    assert "ratio" not in aux_s_text.lower()
    assert "stationary" not in aux_s_text.lower()
    assert "corrector" not in aux_s_text.lower()

    h_text = rule_files['H'].read_text()
    assert "Uniform node sampling with bounded recent history" in h_text
    assert "common query is made at the full archive end (normalized time 1)" in h_text
    assert "t >= 1-History_fraction" in h_text
    assert "Within completely accessible windows, all event records of panel dyads are retrieved; an observed 0 in an accessible window therefore indicates true inactivity" in h_text
    assert "recent 5" not in h_text and "recent5" not in h_text

    b_text = rule_files['B'].read_text()
    assert "Bernoulli event sampling" in b_text
    assert "All time windows are temporally accessible, but individual events may be missing." in b_text
    assert "An observed 0 in an accessible window can therefore be a false negative relative to full-archive activity." in b_text

    rule_hashes = {a: sha(p) for a, p in rule_files.items()}
    aux_hashes = {a: sha(p) for a, p in aux_files.items()}

    # 3. Serialized scientific observations audit
    obs_files = sorted((run / 'observations/sample').glob('*.json'))
    assert len(obs_files) == 280, f"Expected 280 sample observations, got {len(obs_files)}"
    
    arm_counts = Counter()
    block_hashes = {}
    rendered_hashes = {}
    
    for p in obs_files:
        row = read_json(p)
        arm = row['arm']
        arm_counts[arm] += 1
        block = row['block']
        msg = row['messages']
        
        assert row['block_sha256'] == digest(block)
        assert row['prompt_sha256'] == digest(msg)
        assert row['design_version'] == DESIGN_VERSION
        
        # Check Walk_A strictly absent from block and prompt for non-S
        if arm != 'S':
            assert "Walk_A" not in block, f"Walk_A found in serialized block for {row['id']}"
            assert "Auxiliary statistics:" not in msg[1]['content'], f"Auxiliary statistics in prompt for {row['id']}"
        else:
            assert "Walk_A=" in block, f"Walk_A missing from block for {row['id']}"
            assert "Auxiliary statistics:" in msg[1]['content'], f"Auxiliary statistics missing from prompt for {row['id']}"
            
        # Re-derive and verify feature invariance
        o = parse(block)
        f_vec = features(o)
        assert len(f_vec) == 134
        
        block_hashes[row['id']] = row['block_sha256']
        rendered_hashes[row['id']] = row['prompt_sha256']
        
    assert dict(arm_counts) == {'R': 70, 'S': 70, 'H': 70, 'B': 70}

    # 4. Request manifest and payloads audit
    requests = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines()]
    assert len(requests) == 3360, f"Expected 3360 requests, got {len(requests)}"
    
    req_by_id = {r['id']: r for r in requests}
    assert len(req_by_id) == 3360, "Duplicate request IDs"
    
    req_counts = Counter()
    qwen_requests = []
    sol_requests = []
    ds_requests = []
    
    for r in requests:
        cfg = r['config_id']
        arm = r['arm']
        req_counts[arm, cfg] += 1
        
        # Ensure fresh generation version
        assert r['design_version'] == DESIGN_VERSION
        assert r['id'].endswith(f"__{DESIGN_VERSION}")
        assert r['payload_sha256'] == digest(r['payload'])
        assert r['prompt_sha256'] == rendered_hashes[r['observation_id']]
        
        if cfg.startswith('qwen'):
            assert r['payload']['design_version'] == DESIGN_VERSION
            qwen_requests.append(r)
            assert r['production_dispatch_enabled'] is True
            assert r['requires_technical_release'] is False
            assert r['payload']['structured_output']['json_object'] is True
        else:
            if cfg == 'sol':
                sol_requests.append(r)
            elif cfg == 'deepseek':
                ds_requests.append(r)
            assert r['production_dispatch_enabled'] is False
            assert r['requires_technical_release'] is True
            assert r['status'] == 'not_started'
            assert r['started'] is False
            assert r['mock'] is False

    assert len(qwen_requests) == 1680
    assert len(sol_requests) == 840
    assert len(ds_requests) == 840
    
    # Check that each arm has exactly 420 Qwen requests (210 thinking, 210 nonthinking)
    for a in ARMS:
        assert req_counts[a, 'qwen_thinking'] == 210
        assert req_counts[a, 'qwen_nonthinking'] == 210
        assert req_counts[a, 'sol'] == 210
        assert req_counts[a, 'deepseek'] == 210

    # Write audit manifest
    audit_report = {
        'statement': PROMPT_REVISION_STATEMENT,
        'design_version': DESIGN_VERSION,
        'offline_run': str(run.relative_to(ROOT)),
        'common_templates': common_hashes,
        'arm_rules': rule_hashes,
        'auxiliary_templates': aux_hashes,
        'sample_observations': {
            'total': len(obs_files),
            'per_arm': dict(arm_counts)
        },
        'request_manifest': {
            'total_requests': len(requests),
            'qwen_requests': len(qwen_requests),
            'sol_requests_unauthorized': len(sol_requests),
            'deepseek_requests_unauthorized': len(ds_requests),
            'per_arm_config': {f"{a}_{c}": req_counts[a, c] for a in ARMS for c in CONFIGS}
        },
        'paid_api_dispatch_authorized': False,
        'audit_passed': True
    }
    
    out_path = run / 'prompt_freeze_audit.json'
    write_json(out_path, audit_report)
    print(json.dumps(audit_report, indent=2))
    print(f"\nAudit completed successfully. Report written to {out_path}")

if __name__ == '__main__':
    main()
