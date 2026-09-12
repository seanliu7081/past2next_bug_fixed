"""Live-upload this experiment's offline W&B logs without restarting training."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def training_run_id(path):
    """Ignore metadata-only DDP ranks and tolerate partially written records."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    reader = DataStore()
    run_id = None
    try:
        # W&B 0.29's open_for_scan uses r+b; classification only needs rb.
        reader._fname = str(path)
        reader._fp = path.open("rb")
        reader._opened_for_scan = True
        reader._size_bytes = path.stat().st_size
        reader._read_header()
        for _ in range(1024):
            raw = reader.scan_data()
            if raw is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            if record.HasField("run"):
                run_id = record.run.run_id
            if record.HasField("history"):
                keys = {item.key or ".".join(item.nested_key)
                        for item in record.history.item}
                if keys.intersection({"train_loss", "val_loss", "epoch"}):
                    return run_id or path.stem.removeprefix("run-")
    except Exception:
        # The writer may still be completing the final record. Retry next scan.
        return None
    finally:
        if reader._fp is not None:
            reader.close()
    return None


def discover(jobs):
    for job in jobs:
        for stage in ("tokenizer", "policy"):
            folder = Path(job["output_dir"]) / stage / "wandb"
            for path in sorted(folder.glob("offline-run-*/*.wandb")):
                yield job, stage, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="real_robot")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = json.loads((args.run_root / "jobs.json").read_text())
    if args.dry_run:
        for job, stage, path in discover(jobs):
            run_id = training_run_id(path)
            if run_id:
                print(json.dumps({"job": job["name"], "stage": stage,
                                  "file": str(path), "run_id": run_id}))
        return

    folder = args.run_root / "wandb_live_sync"
    folder.mkdir(exist_ok=True)
    state_path = folder / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"runs": {}}
    state.update(entity=args.entity, project=args.project)
    processes = {}
    classify_cache = {}
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = os.environ.copy()
    environment["WANDB_MODE"] = "online"
    environment["WANDB_BASE_URL"] = "https://api.wandb.ai"
    for key in ("WANDB_SERVICE", "WANDB_RUN_ID", "WANDB_RESUME"):
        environment.pop(key, None)
    wandb_cli = str(Path(sys.executable).with_name("wandb"))

    def save():
        state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    try:
        while running:
            for key, (process, log) in list(processes.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    state["runs"][key].update(
                        state=("synced" if code == 0 and Path(key + ".synced").is_file()
                               else "retry_pending"),
                        exit_code=code, retry_after=time.time() + 60)
                    del processes[key]
            for job, stage, path in discover(jobs):
                key = str(path)
                entry = state["runs"].get(key, {})
                if key in processes or entry.get("state") == "synced":
                    continue
                if time.time() < entry.get("retry_after", 0):
                    continue
                fingerprint = (path.stat().st_size, path.stat().st_mtime_ns)
                cached = classify_cache.get(key)
                if cached and cached[0] == fingerprint:
                    run_id = cached[1]
                else:
                    run_id = training_run_id(path)
                    classify_cache[key] = (fingerprint, run_id)
                if not run_id:
                    continue
                url = f"https://wandb.ai/{args.entity}/{args.project}/runs/{run_id}"
                command = [wandb_cli, "beta", "sync", "--live", "--yes",
                           "--entity", args.entity, "--project", args.project,
                           "--job-type", stage, str(path)]
                log_path = folder / f"{job['name']}-{stage}-{run_id}.log"
                log = log_path.open("a")
                process = subprocess.Popen(command, cwd=folder, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT)
                processes[key] = (process, log)
                state["runs"][key] = {"job": job["name"], "stage": stage,
                                      "run_id": run_id, "url": url, "state": "syncing",
                                      "pid": process.pid, "log": str(log_path)}
                print(f"Live sync {job['name']} {stage}: {url}", flush=True)
            save()
            time.sleep(args.poll_seconds)
    finally:
        for key, (process, log) in processes.items():
            if process.poll() is None:
                process.terminate()
            log.close()
            state["runs"][key]["state"] = "stopped"
        save()


if __name__ == "__main__":
    main()
