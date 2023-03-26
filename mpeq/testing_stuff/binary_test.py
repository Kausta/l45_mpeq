import copy

probes = dict()
for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    probes[stage] = {}
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
        probes[stage][loc] = {}

spec = {
    'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'target': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'return': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
    'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
    'low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    'high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    'mid': (Stage.HINT, Location.NODE, Type.MASK_ONE)
}

for name in spec:
    stage, loc, t = spec[name]
    probes[stage][loc][name] = {}
    probes[stage][loc][name]['data'] = []
    probes[stage][loc][name]['type_'] = t
    
A = np.array([2.0, 3, 5, 7, 11])

T_pos = np.arange(A.shape[0])

push(
      probes,
      Stage.INPUT,
      next_probe={
          'pos': np.copy(T_pos) * 1.0 / A.shape[0],
          'key': np.copy(A),
          'target': x
      }
)

push(
      probes,
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(T_pos)),
          'low': mask_one(A.shape[0] - 1, A.shape[0]),
          'high': mask_one(0, A.shape[0]),
          'mid': mask_one((A.shape[0] - 1) // 2, A.shape[0]),
      })

low = 0
high = A.shape[0] - 1

while low < high:
    mid = (low + high) // 2
    if x <= A[mid]:
        high = mid
    else:
        low = mid + 1
        
    push(
        probes,
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(T_pos)),
            'low': mask_one(low, A.shape[0]),
            'high': mask_one(high, A.shape[0]),
            'mid': mask_one((low + high) // 2, A.shape[0]),
        }
    )

push(
    probes,
    Stage.OUTPUT,
    next_probe={'return': mask_one(high, A.shape[0])}
)


finalize(probes)
