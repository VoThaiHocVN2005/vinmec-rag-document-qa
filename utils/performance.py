# utils/performance.py
import time
import psutil
import os

class PerformanceTracker:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.marks = {}
        self.process = psutil.Process(os.getpid())

    def mark(self, name: str):
        self.marks[name] = time.perf_counter()

    def elapsed(self, start, end):
        return self.marks[end] - self.marks[start]

    def total_time(self):
        return time.perf_counter() - self.start_time

    def memory_mb(self):
        return self.process.memory_info().rss / 1024 / 1024

    def summary(self):
        return {
            "total_time_sec": round(self.total_time(), 3),
            "memory_mb": round(self.memory_mb(), 2),
            "steps": {
                k: round(v - self.start_time, 3)
                for k, v in self.marks.items()
            }
        }