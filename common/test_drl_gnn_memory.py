"""Compare streaming PPO against the former whole-minibatch backward pass."""
from copy import deepcopy
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.drl_gnn.config import DRLGNNConfig
from algorithms.drl_gnn.networks import GraphState
from algorithms.drl_gnn.ppo import PPOAgent, Transition, compute_gae
from algorithms.drl_gnn.runner import load_checkpoint
from algorithms.drl_gnn.runtime import configure_cuda_memory, CUDA_ALLOCATOR_MAX_BYTES


def graph(seed):
    rng = np.random.RandomState(seed)
    tasks, vms = 5 + seed % 3, 3 + seed % 2
    mask = rng.uniform(size=(tasks, vms)) > .4
    mask[0, 0] = True
    return GraphState(rng.uniform(.1, 1., (tasks, 8)).astype(np.float32),
                      rng.uniform(.1, 1., (vms, 9)).astype(np.float32),
                      np.array([(i, i + 1) for i in range(tasks - 1)], dtype=np.int64),
                      np.ones_like(mask), mask, tuple(range(tasks)))


def rollout(agent, count):
    states = [graph(i) for i in range(4)]
    records = []
    for i in range(count):
        state = states[i % len(states)]
        action, logp, value = agent.act(state)
        # Exercise both PPO clipping branches, unequal advantages and terminal paths.
        records.append(Transition(state, action, logp + (i % 3 - 1) * .25,
            value + (-1) ** i * .35, -.1 + (i % 11) * .03,
            i % 7 == 0, .15, i % 4))
    return records


def gradients(parameters):
    return torch.cat([p.grad.detach().flatten().clone() for p in parameters])


def bulk_reference(agent, records):
    """Frozen pre-change loss reductions: hold every graph until mean().backward()."""
    cfg = agent.cfg
    advantages, returns = compute_gae(records, cfg.gamma, cfg.gae_lambda)
    old_logp = torch.tensor([r.logp for r in records])
    old_values = torch.tensor([r.value for r in records])
    advantages, returns = torch.tensor(advantages), torch.tensor(returns)
    indices = np.arange(len(records))
    measurements, captured = [], []
    for _ in range(cfg.update_epochs):
        np.random.shuffle(indices)
        for selection in np.array_split(indices, min(cfg.num_minibatches, len(records))):
            outputs = [agent.network.distribution_and_value(records[int(i)].state) for i in selection]
            logp = torch.stack([d.log_prob(torch.tensor(records[int(i)].action))
                                for (d, _), i in zip(outputs, selection)])
            entropy = torch.stack([d.entropy() for d, _ in outputs]).mean()
            values = torch.stack([v for _, v in outputs])
            logratio = logp - old_logp[selection]
            ratio = logratio.exp()
            adv = advantages[selection]
            if cfg.norm_adv:
                std = adv.std() if len(selection) > 1 else adv.new_zeros(())
                adv = (adv - adv.mean()) / (std + 1e-8)
            policy_loss = torch.maximum(-adv * ratio,
                -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            square = (values - returns[selection]).square()
            if cfg.clip_vloss:
                clipped = old_values[selection] + (values - old_values[selection]).clamp(-cfg.clip_coef, cfg.clip_coef)
                value_loss = .5 * torch.maximum(square, (clipped - returns[selection]).square()).mean()
            else:
                value_loss = .5 * square.mean()
            loss = policy_loss - cfg.ent_coef * entropy + cfg.vf_coef * value_loss
            agent.optimizer.zero_grad()
            loss.backward()
            captured.append(gradients(agent.network.parameters()))
            norm = torch.nn.utils.clip_grad_norm_(agent.network.parameters(), cfg.max_grad_norm)
            agent.optimizer.step()
            measurements.append([float(x.detach()) for x in (
                loss, policy_loss, value_loss, entropy, ((ratio - 1) - logratio).mean(),
                (-logratio).mean(), ((ratio - 1).abs() > cfg.clip_coef).float().mean(), norm)])
        if cfg.target_kl is not None and measurements[-1][4] > cfg.target_kl:
            break
    return np.mean(measurements, axis=0), captured


class SavedStorageTracker:
    """Count live nonparameter storages, releasing them with each autograd save."""
    def __init__(self, model):
        self.shared = {t.untyped_storage().data_ptr() for t in (*model.parameters(), *model.buffers())}
        self.references = {}
        self.active = self.peak = 0

    def pack(self, tensor):
        return SavedStorage(tensor, self)


class SavedStorage:
    def __init__(self, tensor, tracker):
        self.tensor = tensor.detach()
        self.tracker = tracker
        storage = tensor.untyped_storage()
        self.key, self.size = storage.data_ptr(), storage.nbytes()
        if self.key not in tracker.shared:
            if self.key not in tracker.references:
                tracker.active += self.size
            tracker.references[self.key] = tracker.references.get(self.key, 0) + 1
            tracker.peak = max(tracker.peak, tracker.active)

    def __del__(self):
        tracker = self.tracker
        if self.key not in tracker.shared:
            tracker.references[self.key] -= 1
            if not tracker.references[self.key]:
                del tracker.references[self.key]
                tracker.active -= self.size


class DRLGNNMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gradients_updates_bn_and_metrics_match_bulk_reference(self):
        configurations = [(512, DRLGNNConfig()),
                          (9, DRLGNNConfig(clip_vloss=False, norm_adv=False)),
                          (1, DRLGNNConfig()),
                          (9, DRLGNNConfig(target_kl=1e-12))]
        for count, cfg in configurations:
            with self.subTest(count=count, clip_vloss=cfg.clip_vloss, target_kl=cfg.target_kl):
                torch.manual_seed(31)
                reference = PPOAgent(cfg)
                records = rollout(reference, count)
                streaming = deepcopy(reference)
                np.random.seed(77)
                # Compare gradients at identical parameters. Separate real-Adam
                # coverage below allows rounding to accumulate across updates.
                with patch.object(reference.optimizer, 'step'):
                    expected, expected_gradients = bulk_reference(reference, records)
                captured = []
                clip = torch.nn.utils.clip_grad_norm_

                def capture(parameters, *args, **kwargs):
                    parameters = list(parameters)
                    captured.append(gradients(parameters))
                    return clip(parameters, *args, **kwargs)

                np.random.seed(77)
                with patch('algorithms.drl_gnn.ppo.nn.utils.clip_grad_norm_', side_effect=capture), \
                     patch.object(streaming.optimizer, 'step') as step, \
                     patch.object(streaming.optimizer, 'zero_grad', wraps=streaming.optimizer.zero_grad) as zero:
                    result = streaming.update(records)
                self.assertEqual(len(captured), len(expected_gradients))
                self.assertEqual(step.call_count, len(expected_gradients))
                self.assertEqual(zero.call_count, len(expected_gradients))
                for actual, expected_gradient in zip(captured, expected_gradients):
                    torch.testing.assert_close(actual, expected_gradient, rtol=5e-4, atol=3e-6)
                for key, value in reference.network.state_dict().items():
                    torch.testing.assert_close(streaming.network.state_dict()[key], value, rtol=5e-4, atol=3e-6)
                keys = ('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'old_approx_kl', 'clipfrac', 'grad_norm')
                np.testing.assert_allclose([result[k] for k in keys], expected, rtol=5e-4, atol=3e-6)
                self.assertEqual(result['epochs'], len(captured) // min(cfg.num_minibatches, count))

    def test_real_adam_updates_preserve_counts_and_stay_numerically_close(self):
        torch.manual_seed(31)
        reference = PPOAgent()
        records = rollout(reference, 512)
        streaming = deepcopy(reference)
        np.random.seed(77)
        expected, expected_gradients = bulk_reference(reference, records)
        captured = []
        clip = torch.nn.utils.clip_grad_norm_

        def capture(parameters, *args, **kwargs):
            parameters = list(parameters)
            captured.append(gradients(parameters))
            return clip(parameters, *args, **kwargs)

        np.random.seed(77)
        with patch('algorithms.drl_gnn.ppo.nn.utils.clip_grad_norm_', side_effect=capture), \
             patch.object(streaming.optimizer, 'step', wraps=streaming.optimizer.step) as step:
            result = streaming.update(records)
        self.assertEqual((result['epochs'], step.call_count, len(captured)), (4, 16, 16))
        for actual, wanted in zip(captured, expected_gradients):
            self.assertLess(float((actual - wanted).norm() / wanted.norm().clamp_min(1e-9)), 1e-4)
        # Adam can amplify roundoff on theoretically zero BN bias gradients.
        for key, value in reference.network.state_dict().items():
            torch.testing.assert_close(streaming.network.state_dict()[key], value, rtol=0, atol=2e-4)
        keys = ('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'old_approx_kl', 'clipfrac', 'grad_norm')
        np.testing.assert_allclose([result[k] for k in keys], expected, rtol=1e-4, atol=3e-6)

    def test_128_graph_minibatch_keeps_only_single_graph_activations(self):
        cfg = DRLGNNConfig(num_envs=1, num_steps=128, num_minibatches=1, update_epochs=1)
        torch.manual_seed(17)
        reference = PPOAgent(cfg)
        records = rollout(reference, 128)
        streaming = deepcopy(reference)
        bulk, stream = SavedStorageTracker(reference.network), SavedStorageTracker(streaming.network)
        with torch.autograd.graph.saved_tensors_hooks(bulk.pack, lambda saved: saved.tensor):
            bulk_reference(reference, records)
        with torch.autograd.graph.saved_tensors_hooks(stream.pack, lambda saved: saved.tensor):
            streaming.update(records)
        gc.collect()
        self.assertEqual((bulk.active, stream.active), (0, 0))
        self.assertGreater(bulk.peak, stream.peak * 50)

    def test_cuda_budget_uses_fraction_and_absolute_ceiling(self):
        for gib in (8, 32, 48, 80):
            with self.subTest(gib=gib), \
                 patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=gib * 1024**3)), \
                 patch('torch.cuda.set_per_process_memory_fraction') as limit:
                result = configure_cuda_memory('cuda:0')
                expected = min(.85, 28 / gib)
                limit.assert_called_once_with(expected, device=torch.device('cuda:0'))
                self.assertLessEqual(result['allocator_limit_bytes'], CUDA_ALLOCATOR_MAX_BYTES)
                self.assertEqual(result['allocator_limit_bytes'], int(gib * 1024**3 * expected))
        with patch('torch.cuda.current_device', return_value=2), \
             patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=32 * 1024**3)), \
             patch('torch.cuda.set_per_process_memory_fraction') as limit:
            configure_cuda_memory('cuda')
            limit.assert_called_once_with(.85, device=torch.device('cuda:2'))
        with patch('torch.cuda.set_per_process_memory_fraction') as limit:
            self.assertIsNone(configure_cuda_memory('cpu')['allocator_limit_bytes'])
            limit.assert_not_called()

    def test_checkpoint_deserializes_on_cpu_even_for_cuda_evaluation(self):
        with patch('algorithms.drl_gnn.runner.torch.load', return_value={}) as load:
            with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                load_checkpoint('unused.pth', None, device='cuda')
            load.assert_called_once_with('unused.pth', map_location='cpu', weights_only=False)


if __name__ == '__main__':
    unittest.main()
