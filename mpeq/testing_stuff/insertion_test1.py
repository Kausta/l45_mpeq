import copy

probes = dict()
for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    probes[stage] = {}
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
        probes[stage][loc] = {}

spec = {
    'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'pred': (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
    'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
    'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
}

for name in spec:
    stage, loc, t = spec[name]
    probes[stage][loc][name] = {}
    probes[stage][loc][name]['data'] = []
    probes[stage][loc][name]['type_'] = t
    
A = np.array([5, 2, 4, 1, 3]) 
A_pos = np.arange(A.shape[0])

push(
      probes,
      Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'key': np.copy(A)
      })

push(
      probes,
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'i': mask_one(0, A.shape[0]),
          'j': mask_one(0, A.shape[0])
      })

# LOOP
for j in range(1, A.shape[0]):
    key = A[j]
    # Insert A[j] into the sorted sequence A[1 .. j - 1]
    i = j - 1
    while i >= 0 and A[i] > key:
      A[i + 1] = A[i]
      A_pos[i + 1] = A_pos[i]
      i -= 1
    A[i + 1] = key
    stor_pos = A_pos[i + 1]
    A_pos[i + 1] = j
    print('A', A)
    print('A_pos', A_pos)
    
    push(
            probes,
            Stage.HINT,
            next_probe={
                'pred_h': array(np.copy(A_pos)),
                'i': mask_one(stor_pos, np.copy(A.shape[0])),
                'j': mask_one(j, np.copy(A.shape[0]))
            })
# end of loop

push(
      probes,
      Stage.OUTPUT,
      next_probe={'pred': array(np.copy(A_pos))})
      
A_pos

finalize(probes)
