import subprocess
import time
from datetime import datetime
import os

def run_experiment(strategy, interval):
    """Run training with specific swapping configuration"""
    cmd = [
        "python3", "train.py",  # Changed to python3
        "--tkn-dim", "128",
        "--qk-dim", "64",
        "--nheads", "8",
        "--hn-mult", "4.0",
        "--attn-beta", "0.125",
        "--time-steps", "12",
        "--blocks", "6",
        "--epochs", "250",
        "--batch-size", "256",
        "--lr", "1e-4",
        "--swap-strategy", str(strategy),
        "--swap-interval", str(interval)
    ]
    
    log_dir = "swapping_experiments"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"strategy_{strategy}_interval_{interval}_{timestamp}.log")
    
    print(f"\n🚀 Starting experiment: strategy={strategy}, interval={interval}")
    print(" ".join(cmd))
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                f.write(output)
                
        rc = process.poll()
        print(f"✅ Experiment completed with exit code {rc}" if rc == 0 else f"❌ Experiment failed with exit code {rc}")

def main():
    # Run baseline first
    run_experiment(0, 0)
    
    # Then run strategies
    strategies = [1, 2, 3, 4]
    intervals = [10, 5, 2, 1]
    
    for strategy in strategies:
        for interval in intervals:
            run_experiment(strategy, interval)
            time.sleep(30)

if __name__ == "__main__":
    main()
