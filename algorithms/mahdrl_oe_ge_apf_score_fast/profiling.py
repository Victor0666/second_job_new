"""Sampled CPU exclusive wall times and deferred CUDA event times, without RNG."""
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
import torch


class StageProfiler:
    def __init__(self, mode="sampled", every=100):
        if mode not in ("off", "sampled") or every < 1:
            raise ValueError("Invalid profiler configuration")
        self.mode, self.every = mode, every
        self.active = False
        self.reset()

    def reset(self):
        self.cpu = defaultdict(lambda: [0, 0.])
        self.events = []
        self.stack = []

    def select(self, index):
        self.active = self.mode == "sampled" and (index == 0 or (index + 1) % self.every == 0)

    @contextmanager
    def stage(self, name, device=None):
        if not self.active:
            yield
            return
        gpu = device is not None and torch.device(device).type == "cuda"
        start_event = end_event = None
        if gpu:
            with torch.cuda.device(device):
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
        frame = [perf_counter(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            self.cpu[name][0] += 1
            self.cpu[name][1] += max(0., elapsed - frame[1])
            if gpu:
                with torch.cuda.device(device):
                    end_event.record()
                self.events.append((name, device, start_event, end_event))

    def report(self):
        for device in {str(e[1]) for e in self.events}:
            torch.cuda.synchronize(torch.device(device))
        gpu = defaultdict(lambda: [0, 0.])
        for name, _, start, end in self.events:
            gpu[name][0] += 1
            gpu[name][1] += start.elapsed_time(end) / 1000.
        result = dict(mode=self.mode, every=self.every, scope="first_and_every_N_updates_or_decisions_no_extrapolation",
                      cpu_exclusive={k: dict(calls=v[0], seconds=v[1]) for k, v in self.cpu.items()},
                      cuda_elapsed={k: dict(calls=v[0], seconds=v[1]) for k, v in gpu.items()},
                      timing_note="CPU wall and CUDA elapsed overlap; do not add them together")
        self.reset()
        return result
