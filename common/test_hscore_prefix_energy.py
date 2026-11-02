"""Differential accounting across boundaries, idle gaps and fractional powers."""
import random
import unittest
from common.resource_opt import Host, _power_model_factory
from common.workflow_opt import LoadRecord, energy_from_records
from algorithms.mahdrl_as_fast_hscore_speed2.energy import PrefixEnergy


class PrefixEnergyTests(unittest.TestCase):
    def test_matches_reference_at_every_query(self):
        rng = random.Random(4)
        hosts = {i: Host(i, power_model=_power_model_factory(i), total_pc=16.) for i in range(3)}
        records = []
        ledger = PrefixEnergy(hosts, records, 2.)
        now = 0.
        for step in range(300):
            now += rng.choice((0., .1, .25, 1., 2.))
            if rng.random() < .8:
                records.append(LoadRecord(now, now+rng.choice((0., .1, 1., 4.)),
                    rng.randrange(3), rng.choice((0., .1, .3, 1., 4.))))
            if now < 2.:
                continue
            clipped = [LoadRecord(r.start_time, min(r.end_time, now), r.server_id, r.vm_pc)
                       for r in records if r.end_time > 0 and r.start_time < now
                       and min(r.end_time, now) > r.start_time]
            actual, per_host = ledger.at(now)
            expected = energy_from_records(clipped, hosts, start_time=2., end_time=now)
            self.assertEqual(actual, expected, (step, now))
            for hid in hosts:
                self.assertEqual(per_host[hid], energy_from_records(
                    [r for r in clipped if r.server_id == hid], {hid: hosts[hid]},
                    start_time=2., end_time=now), (step, hid))

    def test_rewind_historical_append_and_idle_window_reopening(self):
        hosts = {0: Host(0, power_model=_power_model_factory(0), total_pc=16.)}
        records = [LoadRecord(0., 1., 0, .1), LoadRecord(5., 8., 0, .3)]
        ledger = PrefixEnergy(hosts, records, 0.)
        for now in (0., 1., 2., 5., 6., 10., 4., 7., 10.):
            if now == 7.:
                records.append(LoadRecord(2., 9., 0, .2))
            clipped = [LoadRecord(r.start_time, min(r.end_time, now), r.server_id, r.vm_pc)
                       for r in records if r.start_time < now and min(r.end_time, now) > r.start_time]
            self.assertEqual(ledger.at(now)[0], energy_from_records(
                clipped, hosts, start_time=0., end_time=now))
        records.clear()
        self.assertEqual(ledger.at(11.), (0., {0: 0.}))


if __name__ == '__main__':
    unittest.main()
