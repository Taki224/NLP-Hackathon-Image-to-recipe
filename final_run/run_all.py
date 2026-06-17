import subprocess
import time
import datetime
import os
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).parent.parent
FINAL_RUN_DIR = PROJECT_ROOT / 'final_run'
REPORTS_DIR = PROJECT_ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

REPORT_FILE = FINAL_RUN_DIR / "training_report.md"
LOG_FILE = FINAL_RUN_DIR / "run.log"

SCRIPTS_TO_RUN = [
    {"name": "Data Splitting (No Leakage)", "script": "split_data.py"},
    {"name": "Phase 1 (Baseline 5k)", "script": "train_phase1.py"},
    {"name": "Phase 2 (Precision 8k)", "script": "train_phase2_8k.py"},
    {"name": "Phase 2 (Scale 74k)", "script": "train_phase2_74k.py"},
    {"name": "DAR Augmentation (74k)", "script": "data_augmentation.py"},
    {"name": "Phase 3 DAR (74k)", "script": "train_dar_74k.py"},
    {"name": "Evaluation (Recall@K)", "script": "evaluate.py"},
    {"name": "Visualization (t-SNE)", "script": "visualize.py"},
]

def write_log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

def write_report_header():
    with open(REPORT_FILE, "w") as f:
        f.write("# Overnight Run Report\n\n")

def append_to_report(msg):
    with open(REPORT_FILE, "a") as f:
        f.write(msg + "\n")

def get_latest_loss(log_dir_name):
    base_dir = REPORTS_DIR / log_dir_name
    if not base_dir.exists():
        return "N/A (No logs found)"
    
    versions = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith("version_"):
            try:
                v_num = int(d.name.split("_")[1])
                versions.append((v_num, d))
            except ValueError:
                pass
    
    if not versions:
        return "N/A (No versions found)"
    
    latest_version_dir = sorted(versions)[-1][1]
    metrics_path = latest_version_dir / "metrics.csv"
    if not metrics_path.exists():
        return f"N/A (No metrics.csv in {latest_version_dir.name})"
    
    last_loss = "N/A"
    try:
        with open(metrics_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'train_loss_epoch' in row and row['train_loss_epoch']:
                    last_loss = row['train_loss_epoch']
                elif 'train_loss' in row and row['train_loss']:
                    last_loss = row['train_loss']
    except Exception as e:
        return f"Error reading csv: {e}"
    return last_loss

def run_script(script_info):
    name = script_info["name"]
    script = script_info["script"]
    script_path = FINAL_RUN_DIR / script
    
    write_log(f"Starting {name} ({script})")
    start_time = time.time()
    
    max_retries = 3
    success = False
    
    for attempt in range(1, max_retries + 1):
        try:
            write_log(f"Attempt {attempt}/{max_retries} for {name}")
            
            # Subprocess streams output directly to terminal so you can see it live
            cmd = ["uv", "run"]
            if script == "visualize.py":
                cmd.extend(["--with", "scikit-learn"])
            cmd.extend(["python3", str(script_path)])
            
            result = subprocess.run(
                cmd,
                cwd=str(FINAL_RUN_DIR),
                check=True
            )
            success = True
            write_log(f"{name} completed successfully on attempt {attempt}.")
            break
        except subprocess.CalledProcessError as e:
            write_log(f"Error: {name} failed with exit code {e.returncode} on attempt {attempt}.")
            if attempt < max_retries:
                write_log("Waiting 30 seconds before autorestart...")
                time.sleep(30)
            else:
                write_log(f"{name} exhausted all retries. Moving to next script.")
    
    end_time = time.time()
    duration_secs = end_time - start_time
    duration_str = str(datetime.timedelta(seconds=int(duration_secs)))
    
    log_dir_name = "lightning_logs"
    if script == "train_dar_74k.py":
        log_dir_name = "dar_lightning_logs"
        
    final_loss = "N/A"
    if "train" in script:
        final_loss = get_latest_loss(log_dir_name)
    
    status_str = "✅ SUCCESS" if success else "❌ FAILED"
    
    report_entry = f"### {name}\n"
    report_entry += f"- **Status**: {status_str}\n"
    report_entry += f"- **Duration**: {duration_str}\n"
    if "train" in script:
        report_entry += f"- **Final Loss**: {final_loss}\n"
    report_entry += "\n"
    
    append_to_report(report_entry)

if __name__ == "__main__":
    write_log("=== STARTING OVERNIGHT PIPELINE ===")
    write_report_header()
    
    for script_info in SCRIPTS_TO_RUN:
        run_script(script_info)
        
    write_log("=== PIPELINE COMPLETE ===")
