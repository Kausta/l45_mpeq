import numpy as np
from typing import Dict, Union, List

class Stage:
    INPUT = 'input'
    OUTPUT = 'output'
    HINT = 'hint'

class Location:
    NODE = 'node'
    EDGE = 'edge'
    GRAPH = 'graph'
  
class Type:
    SCALAR = 'scalar'
    CATEGORICAL = 'categorical'
    MASK = 'mask'
    MASK_ONE = 'mask_one'
    POINTER = 'pointer'
    SHOULD_BE_PERMUTATION = 'should_be_permutation'
    PERMUTATION_POINTER = 'permutation_pointer'
    SOFT_POINTER = 'soft_pointer'
 
class OutputClass:
    POSITIVE = 1
    NEGATIVE = 0
    MASKED = -1


_Location = Location
_Stage = Stage
_Type = Type
_OutputClass = OutputClass

_Array = np.ndarray
_Data = Union[_Array, List[_Array]]
_DataOrType = Union[_Data, str]

ProbesDict = Dict[
    str, Dict[str, Dict[str, Dict[str, _DataOrType]]]]


def push(probes: ProbesDict, stage: str, next_probe):
  """Pushes a probe into an existing `ProbesDict`."""
  for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
    for name in probes[stage][loc]:
      if name not in next_probe:
        raise ProbeError(f'Missing probe for {name}.')
      if isinstance(probes[stage][loc][name]['data'], _Array):
        raise ProbeError('Attemping to push to finalized `ProbesDict`.')
      # Pytype thinks initialize() returns a ProbesDict with a str for all final
      # values instead of _DataOrType.
      probes[stage][loc][name]['data'].append(next_probe[name])  # pytype: disable=attribute-error
      
def array(A_pos: np.ndarray) -> np.ndarray:
  """Constructs an `array` probe."""
  probe = np.arange(A_pos.shape[0])
  for i in range(1, A_pos.shape[0]):
    probe[A_pos[i]] = A_pos[i - 1]
  return probe

def mask_one(i: int, n: int) -> np.ndarray:
  """Constructs a `mask_one` probe."""
  assert n > i
  probe = np.zeros(n)
  probe[i] = 1
  return probe

def finalize(probes: ProbesDict):
  """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
  for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
      for name in probes[stage][loc]:
        # if isinstance(probes[stage][loc][name]['data'], _Array):
        #   raise ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
        if stage == _Stage.HINT:
          # Hints are provided for each timestep. Stack them here.
          probes[stage][loc][name]['data'] = np.stack(
              probes[stage][loc][name]['data'])
        else:
          # Only one instance of input/output exist. Remove leading axis.
          probes[stage][loc][name]['data'] = np.squeeze(
              np.array(probes[stage][loc][name]['data']))

