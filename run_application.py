import signal
import subprocess
import sys
import time


processes = []


def stop_processes():
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def handle_signal(signum, _frame):
    stop_processes()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

processes.append(subprocess.Popen([sys.executable, "run_api_service.py"]))
processes.append(
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "run_dashboard.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
        ]
    )
)

while True:
    for process in processes:
        exit_code = process.poll()
        if exit_code is not None:
            stop_processes()
            raise SystemExit(exit_code or 1)
    time.sleep(0.5)
