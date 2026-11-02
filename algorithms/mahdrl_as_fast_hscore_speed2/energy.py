"""Cache settled host event terms, retaining the original summation order."""
import heapq
import numpy as np


class _HostPrefix:
    def __init__(self, host):
        self.host = host
        self.pending, self.events, self.ends = [], {}, {}
        self.terms = []
        self.load = self.energy = 0.
        self.previous = None
        self.right = -float('inf')

    def event(self, when, delta):
        if when not in self.events:
            self.events[when] = 0.
            heapq.heappush(self.pending, when)
        self.events[when] += delta

    def add(self, record, origin):
        start, end, pc = float(record.start_time), float(record.end_time), float(record.vm_pc)
        if pc <= 0 or end <= start or end <= 0:
            return
        start = max(origin, start)
        self.ends[start] = max(self.ends.get(start, -float('inf')), end)
        self.event(start, pc if end > start else 0.)
        if end > start:
            self.event(end, -pc)

    def power(self):
        ratio = max(0., min(1., self.load / max(float(self.host.total_pc), 1e-9)))
        return float(self.host.power(ratio))

    def at(self, now):
        # Starts at exactly `now` may still be appended; keep that boundary unsettled.
        while self.pending and self.pending[0] < now:
            point = heapq.heappop(self.pending)
            self.right = max(self.right, self.ends.pop(point, -float('inf')))
            if self.previous is None:
                self.previous = point
            term = self.power() * (point-self.previous)
            self.terms.append(term)
            self.energy += term
            self.load += self.events.pop(point)
            self.previous = point
        if self.previous is None:
            return 0., 0.
        tail = self.power() * max(0., min(now, self.right)-self.previous)
        return self.energy+tail, tail


class PrefixEnergy:
    def __init__(self, hosts, records, origin):
        self.hosts, self.records, self.origin = hosts, records, float(origin)
        self.prefix = {hid: _HostPrefix(host) for hid, host in hosts.items()}
        self.count = 0
        self.time = -float('inf')

    def at(self, now):
        now = float(now)
        if (now < self.time or len(self.records) < self.count
                or any(r.start_time < self.time for r in self.records[self.count:])):
            # Diagnostic callers may rewind or append historical records.
            self.__init__(self.hosts, self.records, self.origin)
        for record in self.records[self.count:]:
            self.prefix[record.server_id].add(record, self.origin)
        self.count, self.time = len(self.records), now
        terms, per_host = [], {}
        for hid, prefix in self.prefix.items():
            energy, tail = prefix.at(now)
            per_host[hid] = energy
            terms.extend(prefix.terms)
            terms.append(tail)
        # cumsum uses ordered float64 additions; sum would change reduction order.
        total = float(np.cumsum(terms, dtype=np.float64)[-1]) if terms else 0.
        return total, per_host
