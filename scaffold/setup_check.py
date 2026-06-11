"""
setup_check.py
==============

Run this first on any new machine to verify the scaffold is working.

    python setup_check.py

It checks:
  1. All 11 modules import cleanly
  2. Core math (sympy, numpy, scipy) is available
  3. Your LLM API connection works (Claude or OpenAI)
  4. A minimal end-to-end pipeline test runs

Usage:
    python setup_check.py                     # checks imports only
    python setup_check.py --api-key YOUR_KEY  # also tests LLM connection
    python setup_check.py --full              # full end-to-end test
"""

import sys
import os
import argparse
import importlib

# ---- 1. Module imports ----

MODULES = [
    'relational_graph',
    'graph_extractor',
    'structure_mapper',
    'technique_composer',
    'cross_domain_composer',
    'gue_engine',
    'gue_domain_tester',
    'directed_graph',
    'hermitian_gue',
    'metacognition',
    'curiosity_engine',
]

def check_imports():
    print("\n── Module imports ──────────────────────────────────")
    failed = []
    for m in MODULES:
        try:
            importlib.import_module(m)
            print(f"  ✓ {m}")
        except Exception as e:
            print(f"  ✗ {m}: {e}")
            failed.append((m, str(e)))
    if not failed:
        print(f"  All {len(MODULES)} modules OK.")
    return failed

# ---- 2. Scientific libraries ----

def check_libraries():
    print("\n── Scientific libraries ────────────────────────────")
    libs = {
        'numpy':  'import numpy as np; print(f"  ✓ numpy {np.__version__}")',
        'scipy':  'import scipy; print(f"  ✓ scipy {scipy.__version__}")',
        'sympy':  'import sympy; print(f"  ✓ sympy {sympy.__version__}")',
        'z3':     'import z3; print(f"  ✓ z3-solver {z3.get_version_string()}")',
    }
    optional = {'z3'}
    missing_required = []
    for name, check in libs.items():
        try:
            exec(check)
        except ImportError:
            if name in optional:
                print(f"  ○ {name} (optional — install with: pip install {name}-solver)")
            else:
                print(f"  ✗ {name} MISSING — install with: pip install {name}")
                missing_required.append(name)
    return missing_required

# ---- 3. LLM connection ----

def check_llm(api_key=None):
    print("\n── LLM connection ──────────────────────────────────")

    # Try Anthropic Claude
    try:
        import requests
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("  ○ No ANTHROPIC_API_KEY set. Skipping Claude test.")
            print("    Set it with: export ANTHROPIC_API_KEY=your_key")
            print("    Or get one at: https://console.anthropic.com")
            return None

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "Say: OK"}],
            },
            timeout=10,
        )
        if resp.status_code == 200:
            reply = resp.json()["content"][0]["text"].strip()
            print(f"  ✓ Claude API working. Response: '{reply}'")
            return "claude"
        else:
            print(f"  ✗ Claude API error {resp.status_code}: {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"  ✗ Claude connection failed: {e}")
        return None

# ---- 4. Minimal pipeline test ----

def check_pipeline(llm_type=None):
    print("\n── Minimal pipeline test ───────────────────────────")
    import tempfile, shutil
    import numpy as np
    from directed_graph import DirectedCoOccurrenceGraph
    from hermitian_gue import measure_hermitian_gue
    from gue_engine import gue_score, identify_ensemble

    tmp = tempfile.mkdtemp()
    try:
        # Directed graph → Hermitian GUE
        g = DirectedCoOccurrenceGraph(f"{tmp}/test.json")
        for i in range(30):
            g.update_from_run([f"t{i%4}", f"t{(i+1)%4}", f"t{(i+2)%4}"])
        r = measure_hermitian_gue(g, verbose=False)
        print(f"  ✓ Directed graph: {r.n_techniques} techniques, "
              f"Im={r.imaginary_fraction:.2f}, "
              f"ensemble={r.ensemble_identified}")

        # GUE engine
        vals = np.cumsum(np.abs(np.random.randn(50)))
        ens = identify_ensemble(vals)
        print(f"  ✓ GUE engine: identified {ens['closest']}")

        # Structure mapper
        from relational_graph import RelationalGraph, Entity, Relation
        from structure_mapper import map_structures
        g1 = RelationalGraph(domain="test", source_label="source")
        g2 = RelationalGraph(domain="test", source_label="target")
        for g_obj, ids in [(g1, ["a","b","c"]), (g2, ["x","y","z"])]:
            for eid in ids:
                g_obj.add_entity(Entity(id_=eid, type="VARIABLE"))
            g_obj.add_relation(Relation("DEPENDS_ON", (ids[0], ids[1])))
            g_obj.add_relation(Relation("CAUSES", (ids[1], ids[2])))
        m = map_structures(g1, g2)
        print(f"  ✓ Structure mapper: score={m.score:.3f}, "
              f"{m.n_relations_preserved} relations preserved")

        print("  ✓ All pipeline components operational.")
        return True
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmp)

# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║         Scaffold Setup Check                     ║")
    print("╚══════════════════════════════════════════════════╝")

    import_failures = check_imports()
    lib_failures    = check_libraries()
    llm_type        = check_llm(args.api_key)

    if args.full or not import_failures:
        pipeline_ok = check_pipeline(llm_type)
    else:
        pipeline_ok = None

    print("\n── Summary ─────────────────────────────────────────")
    if not import_failures and not lib_failures:
        print("  ✓ All core modules and libraries OK")
    else:
        if import_failures:
            print(f"  ✗ {len(import_failures)} module(s) failed to import")
        if lib_failures:
            print(f"  ✗ Missing required libraries: {lib_failures}")
            print(f"    Fix with: pip install {' '.join(lib_failures)}")

    if llm_type:
        print(f"  ✓ LLM ({llm_type}) connected")
    else:
        print("  ○ No LLM connected yet — set ANTHROPIC_API_KEY")

    if pipeline_ok:
        print("  ✓ Pipeline test passed")

    print()
    if not import_failures and not lib_failures:
        print("Ready. See QUICKSTART.md for next steps.")
    else:
        print("Fix the issues above, then re-run setup_check.py")

if __name__ == "__main__":
    main()
