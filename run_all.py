"""
Runs the whole pipeline in order:  python run_all.py [first_step]
Example: `python run_all.py 4` re-runs steps 4-7 only (steps 1-3 outputs are reused).
Each script can also be run on its own:  python 03_rq1_models.py
"""
import subprocess
import sys

STEPS = ["01_eda.py", "02_features.py", "03_rq1_models.py", "04_rq2_imbalance.py",
         "05_rq3_anomaly.py", "06_powerbi_export.py", "07_report.py"]

start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for script in STEPS[start - 1:]:
    print(f"\n##### Running {script} #####", flush=True)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        sys.exit(f"{script} failed (exit code {result.returncode}); fix it and re-run with: python run_all.py {STEPS.index(script) + 1}")
print("\nAll steps finished.")


