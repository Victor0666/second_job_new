"""Per-process CUDA allocator budget; independent of learning hyperparameters."""
import torch


PPO_UPDATE_EXECUTION = "single_graph_gradient_accumulation_v1"
CUDA_MEMORY_FRACTION = .85
CUDA_ALLOCATOR_MAX_BYTES = 28 * 1024 ** 3


def configure_cuda_memory(device):
    """Leave headroom below 32 GiB; this cannot cap other CUDA users/processes."""
    device = torch.device(device)
    if device.type != 'cuda':
        return dict(device_type=device.type, allocator_limit_bytes=None)
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(CUDA_MEMORY_FRACTION, CUDA_ALLOCATOR_MAX_BYTES / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return dict(device_type='cuda', total_memory_bytes=int(total),
                allocator_fraction=float(fraction), allocator_limit_bytes=int(total * fraction),
                scope='one process PyTorch allocator; excludes external CUDA allocations')


def cuda_memory_usage(device):
    """Allocator observations since process start, not whole-device usage."""
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
