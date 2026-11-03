#!/usr/bin/env bash
# Runs the whole Copilot pipeline in order (same commands as the README Reproduce section, steps 4-14).
set -euo pipefail
python generate_data.py
python src/a1_audit.py > outputs/a1_audit.log
python src/a2_clean_store.py
python src/a3_validate.py | tail -n 1
python src/a4_sql.py > outputs/a4_sql.log
python src/a5_explore.py > outputs/a5_explore.log
python src/b1_segments.py > outputs/b1_segments.log
python src/b2_lapse.py > outputs/b2_lapse.log
python src/c1_triage.py > outputs/triage_run.log
python src/rag.py > outputs/rag_eval.log
python src/agent.py > outputs/agent_run.log
echo "PIPELINE COMPLETE"
