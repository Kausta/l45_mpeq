import abc
from typing import Any, Callable, List, Optional, Tuple

from clrs._src import processors

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["get_processor_factory", "ProcessorFactory", "PGNWithMsg", "PROCESSOR_TAG"]

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'


class PGNWithMsg(processors.PGN):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(
        out_size=out_size,
        mid_size=mid_size,
        mid_act=mid_act,
        activation=activation,
        reduction=reduction,
        msgs_mlp_sizes=msgs_mlp_sizes,
        use_ln=use_ln,
        use_triplets=use_triplets,
        nb_triplet_fts=nb_triplet_fts,
        gated=gated,
        name=name
    )

  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      # Triplet messages, as done by Dudzik and Velickovic (2022)
      triplets = processors.get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)

      o3 = hk.Linear(self.out_size)
      tri_msgs = o3(jnp.max(triplets, axis=1))  # (B, N, N, H)

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    exported_msgs = msgs

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
      gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1-gate)

    
    graph_fts_expanded = jnp.tile(jnp.expand_dims(graph_fts, axis=(1, 2)), reps=(1, n, n, 1))
    node_1 = jnp.tile(jnp.expand_dims(node_fts, axis=1), reps=(1, n, 1, 1))
    node_2 = jnp.tile(jnp.expand_dims(node_fts, axis=2), reps=(1, 1, n, 1))
    msg_input = jnp.concatenate((node_1, node_2, edge_fts, graph_fts_expanded), axis=-1)

    return ret, tri_msgs, exported_msgs, msg_input


ProcessorFactory = Callable[[int], processors.Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_triplet_fts: int,
                          nb_heads: Optional[int] = None) -> ProcessorFactory:
  """Returns a processor factory.

  Args:
    kind: One of the available types of processor.
    use_ln: Whether the processor passes the output through a layernorm layer.
    nb_triplet_fts: How many triplet features to compute.
    nb_heads: Number of attention heads for GAT processors.
  Returns:
    A callable that takes an `out_size` parameter (equal to the hidden
    dimension of the network) and returns a processor instance.
  """
  def _factory(out_size: int, msg_size: int):
    if kind == 'pgn_with_msg':
      processor = PGNWithMsg(
          out_size=out_size,
          # msgs_mlp_sizes=[out_size, out_size],
          msgs_mlp_sizes=[msg_size, msg_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
          mid_size=msg_size
      )
    else:
      raise ValueError('Unexpected processor kind ' + kind)

    return processor

  return _factory

