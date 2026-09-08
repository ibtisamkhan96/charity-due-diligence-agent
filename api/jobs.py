"""A minimal in-memory job store: submit a blocking call, poll for its result.

In-memory and single-process by design, the same honest limitation as this
agent's sibling project. A real multi-instance deployment would need a real
queue (Celery/RQ) and shared storage (Redis) behind this instead.
"""
import threading
import uuid


class JobStore:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create(self):
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {"status": "pending", "log": [], "result": None, "error": None}
        return job_id

    def run_async(self, job_id, fn, kwargs):
        def _run():
            with self._lock:
                self._jobs[job_id]["status"] = "running"
            try:
                result = fn(kwargs)
                with self._lock:
                    self._jobs[job_id]["status"] = "completed"
                    self._jobs[job_id]["result"] = result
                    self._jobs[job_id]["log"] = result.get("log", [])
            except Exception as e:
                with self._lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(e)

        threading.Thread(target=_run, daemon=True).start()

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None
