"""Shared GPU-slot job orchestrator for all FOMO26 downstream finetune runs
(Task 1 clf mini-run, and future Task 2/4 seg runs once FomoStudentSegNet
exists). Factored out after the stratified-10 Task-1 run exposed a real bug
in the per-run copy of this logic: the stall watchdog's `p.kill()` only
killed the `bash -c "..."` wrapper process, not the actual `asp_finetune_*`
child bash spawns via `&&`-chaining (no automatic exec/tail-call) -- so 3
killed-for-stalling jobs left orphaned, still-hung `asp_finetune_cls`
processes running under init (ppid=1) for hours, quietly holding ~2-3.5GB of
GPU memory each until found and killed manually post-hoc.

Fix: launch every job in its own process group (`start_new_session=True`)
and kill the *whole group* (`os.killpg`) on stall, so bash + whatever it
forked all die together. Use this module's `run_orchestrator()` for every
new dataset/task orchestrator going forward instead of copy-pasting the
launch/reap/watchdog loop.
"""
import os
import signal
import subprocess
import time


def available_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


class JobOrchestrator:
    def __init__(self, jobs, build_cmd, scratch_dir, num_gpus=4, jobs_per_gpu=1,
                 min_available_mb=40_000, stall_timeout_s=900, launch_stagger_s=30,
                 poll_interval_s=30):
        """jobs: list of dicts, each must have a unique 'name' key.
        build_cmd: callable(job, gpu) -> shell command string.
        """
        os.makedirs(scratch_dir, exist_ok=True)
        self.queue = list(jobs)
        self.build_cmd = build_cmd
        self.scratch_dir = scratch_dir
        self.num_gpus = num_gpus
        self.jobs_per_gpu = jobs_per_gpu
        self.min_available_mb = min_available_mb
        self.stall_timeout_s = stall_timeout_s
        self.launch_stagger_s = launch_stagger_s
        self.poll_interval_s = poll_interval_s
        self.running = {}  # gpu -> list of entry dicts

    def _log_size(self, path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def _gpu_slot_count(self, gpu):
        return len(self.running.get(gpu, []))

    def _kill_group(self, entry):
        try:
            os.killpg(os.getpgid(entry["popen"].pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # already gone

    def _reap_and_watchdog(self, gpu):
        still = []
        now = time.time()
        for entry in self.running.get(gpu, []):
            p = entry["popen"]
            rc = p.poll()
            if rc is not None:
                status = "OK" if rc == 0 else f"FAILED(exit={rc})"
                print(f"[done] {entry['name']} (gpu{gpu}) {status}", flush=True)
                if rc != 0:
                    self.queue.append(entry["job"])
                continue

            size = self._log_size(entry["log_path"])
            if size != entry["last_size"]:
                entry["last_size"] = size
                entry["last_change_t"] = now
            elif now - entry["last_change_t"] > self.stall_timeout_s:
                mins = int((now - entry["last_change_t"]) / 60)
                print(f"[STALLED] {entry['name']} (gpu{gpu}) no log growth for {mins}min "
                      f"-- killing process group and re-queueing", flush=True)
                self._kill_group(entry)
                time.sleep(2)
                self.queue.append(entry["job"])
                continue

            still.append(entry)
        self.running[gpu] = still

    def _launch(self, job, gpu):
        log_path = os.path.join(self.scratch_dir, job["name"] + ".log")
        cmd = self.build_cmd(job, gpu)
        f = open(log_path, "w")
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT,
                              start_new_session=True)  # own process group -> killpg reaches all children
        now = time.time()
        self.running.setdefault(gpu, []).append({
            "name": job["name"], "job": job, "popen": p, "log_path": log_path,
            "last_size": 0, "last_change_t": now,
        })
        print(f"[launch] {job['name']} -> gpu{gpu} pid={p.pid} avail={available_mb()}MB log={log_path}", flush=True)

    def run(self):
        print(f"total jobs to launch: {len(self.queue)}", flush=True)
        while self.queue or any(self._gpu_slot_count(g) > 0 for g in range(self.num_gpus)):
            for gpu in range(self.num_gpus):
                self._reap_and_watchdog(gpu)
                while (self.queue and self._gpu_slot_count(gpu) < self.jobs_per_gpu
                       and available_mb() > self.min_available_mb):
                    self._launch(self.queue.pop(0), gpu)
                    time.sleep(self.launch_stagger_s)
            time.sleep(self.poll_interval_s)
        print("ALL JOBS COMPLETE", flush=True)


def run_orchestrator(jobs, build_cmd, scratch_dir, **kwargs):
    JobOrchestrator(jobs, build_cmd, scratch_dir, **kwargs).run()
