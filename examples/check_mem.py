from torch.cuda.memory import _record_memory_history, _dump_snapshot
from torch import zeros, complex64, cat
from torch.cuda import empty_cache

_record_memory_history(True, trace_alloc_max_entries=100000, 
                       trace_alloc_record_context=True)

t = [None] * 6
t[0] = zeros(8, 500, 500, dtype=complex64, device='cuda')
t[1] = zeros(8, 500, 500, dtype=complex64, device='cuda')

s = cat([i for i in t if i is not None], dim=0)
print(s.shape)
del s
# s = t[0].clone()

# for tt in t:
#     if tt is not None:
#         del tt

# del s

_dump_snapshot(f"logs/mem_snapshot.pickle")
_record_memory_history(enabled=None)
