import functools
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import baselines
from clrs._src import decoders
from clrs._src import losses
from clrs._src import model
from clrs._src import nets
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from .net_msg import NetMsg

__all__ = ["BaselineMsgModel"]

_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Feedback = samplers.Feedback
_Location = specs.Location
_Seed = jnp.ndarray
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass


class BaselineMsgModel(baselines.BaselineModel):
    """Model implementation with selectable message passing algorithm."""

    def __init__(
        self,
        spec: Union[_Spec, List[_Spec]],
        dummy_trajectory: Union[List[_Feedback], _Feedback],
        processor_factory: processors.ProcessorFactory,
        hidden_dim: int = 32,
        encode_hints: bool = False,
        decode_hints: bool = True,
        encoder_init: str = 'default',
        use_lstm: bool = False,
        learning_rate: float = 0.005,
        grad_clip_max_norm: float = 0.0,
        checkpoint_path: str = '/tmp/clrs3',
        freeze_processor: bool = False,
        dropout_prob: float = 0.0,
        hint_teacher_forcing: float = 0.0,
        hint_repred_mode: str = 'soft',
        name: str = 'base_model',
        nb_msg_passing_steps: int = 1,
        l1_weight: float = 1.0,
    ):
        super().__init__(
            spec=spec,
            dummy_trajectory=dummy_trajectory,
            processor_factory=processor_factory,
            hidden_dim=hidden_dim,
            encode_hints=encode_hints,
            decode_hints=decode_hints,
            encoder_init=encoder_init,
            use_lstm=use_lstm,
            learning_rate=learning_rate,
            grad_clip_max_norm=grad_clip_max_norm,
            checkpoint_path=checkpoint_path,
            freeze_processor=freeze_processor,
            dropout_prob=dropout_prob,
            hint_teacher_forcing=hint_teacher_forcing,
            hint_repred_mode=hint_repred_mode,
            name=name,
            nb_msg_passing_steps=nb_msg_passing_steps
        )

        self.l1_weight = l1_weight

    def _create_net_fns(self, hidden_dim, encode_hints, processor_factory,
                        use_lstm, encoder_init, dropout_prob,
                        hint_teacher_forcing, hint_repred_mode):
        def _use_net(*args, **kwargs):
            return NetMsg(self._spec, hidden_dim, encode_hints, self.decode_hints,
                            processor_factory, use_lstm, encoder_init,
                            dropout_prob, hint_teacher_forcing,
                            hint_repred_mode,
                            self.nb_dims, self.nb_msg_passing_steps)(*args, **kwargs)

        self.net_fn = hk.transform(_use_net)
        pmap_args = dict(axis_name='batch', devices=jax.local_devices())
        n_devices = jax.local_device_count()
        func, static_arg, extra_args = (
            (jax.jit, 'static_argnums', {}) if n_devices == 1 else
            (jax.pmap, 'static_broadcasted_argnums', pmap_args))
        pmean = functools.partial(jax.lax.pmean, axis_name='batch')
        self._maybe_pmean = pmean if n_devices > 1 else lambda x: x
        extra_args[static_arg] = 3
        self.jitted_grad = func(self._compute_grad, **extra_args)
        extra_args[static_arg] = 4
        self.jitted_feedback = func(self._feedback, donate_argnums=[0, 3],
                                    **extra_args)
        extra_args[static_arg] = [3, 4, 5]
        self.jitted_predict = func(self._predict, **extra_args)
        extra_args[static_arg] = [3, 4]
        self.jitted_accum_opt_update = func(baselines.accum_opt_update, donate_argnums=[0, 2],
                                            **extra_args)

    def _predict(self, params, rng_key: hk.PRNGSequence, features: _Features,
                 algorithm_index: int, return_hints: bool,
                 return_all_outputs: bool):
        outs, hint_preds, all_msgs = self.net_fn.apply(
            params, rng_key, [features],
            repred=True, algorithm_index=algorithm_index,
            return_hints=return_hints,
            return_all_outputs=return_all_outputs)
        outs = decoders.postprocess(self._spec[algorithm_index],
                                    outs,
                                    sinkhorn_temperature=0.1,
                                    sinkhorn_steps=50,
                                    hard=True,
                                    )
        return outs, hint_preds, all_msgs

    def _loss(self, params, rng_key, feedback, algorithm_index):
        """Calculates model loss f(feedback; params)."""
        output_preds, hint_preds, all_msgs = self.net_fn.apply(
            params, rng_key, [feedback.features],
            repred=False,
            algorithm_index=algorithm_index,
            return_hints=True,
            return_all_outputs=False)

        nb_nodes = baselines._nb_nodes(feedback, is_chunked=False)
        lengths = feedback.features.lengths
        total_loss = 0.0

        # Calculate output loss.
        for truth in feedback.outputs:
            total_loss += losses.output_loss(
                truth=truth,
                pred=output_preds[truth.name],
                nb_nodes=nb_nodes,
            )

        # Optionally accumulate hint losses.
        if self.decode_hints:
            for truth in feedback.features.hints:
                total_loss += losses.hint_loss(
                    truth=truth,
                    preds=[x[truth.name] for x in hint_preds],
                    lengths=lengths,
                    nb_nodes=nb_nodes,
                )

        # all_msgs.shape = (nlayers, nbatch, nnodes, nnodes, nmsg_dim)
        total_loss += self.l1_weight * jnp.mean(
            jnp.sum(jnp.abs(all_msgs), axis=(-2, -1))
        )

        return total_loss
