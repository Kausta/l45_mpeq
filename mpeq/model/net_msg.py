import functools

from typing import Dict, List, Optional, Tuple

import chex

from clrs._src import decoders
from clrs._src import encoders
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs
from clrs._src import nets

import haiku as hk
import jax
import jax.numpy as jnp

_Location = specs.Location

__all__ = ["NetMsg"]

@chex.dataclass
class _MessagePassingWithMsgScanState:
    hint_preds: chex.Array
    output_preds: chex.Array
    hiddens: chex.Array
    lstm_state: Optional[hk.LSTMState]
    msgs: chex.Array
    input_msg: chex.Array
    input_algo: chex.Array


class NetMsg(nets.Net):
    def __init__(self,
                 spec: List[nets._Spec],
                 hidden_dim: int,
                 msg_dim: int,
                 encode_hints: bool,
                 decode_hints: bool,
                 processor_factory: processors.ProcessorFactory,
                 use_lstm: bool,
                 encoder_init: str,
                 dropout_prob: float,
                 hint_teacher_forcing: float,
                 hint_repred_mode='soft',
                 nb_dims=None,
                 nb_msg_passing_steps=1,
                 name: str = 'net'):
        super().__init__(spec=spec,
                         hidden_dim=hidden_dim,
                         encode_hints=encode_hints,
                         decode_hints=decode_hints,
                         processor_factory=processor_factory,
                         use_lstm=use_lstm,
                         encoder_init=encoder_init,
                         dropout_prob=dropout_prob,
                         hint_teacher_forcing=hint_teacher_forcing,
                         hint_repred_mode=hint_repred_mode,
                         nb_dims=nb_dims,
                         nb_msg_passing_steps=nb_msg_passing_steps,
                         name=name)
                         
        self.msg_dim = msg_dim

    def _msg_passing_step(self,
                          mp_state: _MessagePassingWithMsgScanState,
                          i: int,
                          hints: List[nets._DataPoint],
                          repred: bool,
                          lengths: chex.Array,
                          batch_size: int,
                          nb_nodes: int,
                          inputs: nets._Trajectory,
                          first_step: bool,
                          spec: nets._Spec,
                          encs: Dict[str, List[hk.Module]],
                          decs: Dict[str, Tuple[hk.Module]],
                          return_hints: bool,
                          return_all_outputs: bool
                          ):
        if self.decode_hints and not first_step:
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            decoded_hint = decoders.postprocess(spec,
                                                mp_state.hint_preds,
                                                sinkhorn_temperature=0.1,
                                                sinkhorn_steps=25,
                                                hard=hard_postprocess)
        if repred and self.decode_hints and not first_step:
            cur_hint = []
            for hint in decoded_hint:
                cur_hint.append(decoded_hint[hint])
        else:
            cur_hint = []
            needs_noise = (self.decode_hints and not first_step and
                           self._hint_teacher_forcing < 1.0)
            if needs_noise:
                # For noisy teacher forcing, choose which examples in the batch to force
                force_mask = jax.random.bernoulli(
                    hk.next_rng_key(), self._hint_teacher_forcing,
                    (batch_size,))
            else:
                force_mask = None
            for hint in hints:
                hint_data = jnp.asarray(hint.data)[i]
                _, loc, typ = spec[hint.name]
                if needs_noise:
                    if (typ == nets._Type.POINTER and
                            decoded_hint[hint.name].type_ == nets._Type.SOFT_POINTER):
                        # When using soft pointers, the decoded hints cannot be summarised
                        # as indices (as would happen in hard postprocessing), so we need
                        # to raise the ground-truth hint (potentially used for teacher
                        # forcing) to its one-hot version.
                        hint_data = hk.one_hot(hint_data, nb_nodes)
                        typ = nets._Type.SOFT_POINTER
                    hint_data = jnp.where(nets._expand_to(force_mask, hint_data),
                                          hint_data,
                                          decoded_hint[hint.name].data)
                cur_hint.append(
                    probing.DataPoint(
                        name=hint.name, location=loc, type_=typ, data=hint_data))

        hiddens, output_preds_cand, hint_preds, lstm_state, msgs_state, input_msg_state, input_algo_state = self._one_step_pred(
            inputs, cur_hint, mp_state.hiddens,
            batch_size, nb_nodes, mp_state.lstm_state,
            spec, encs, decs, repred)

        if first_step:
            output_preds = output_preds_cand
        else:
            output_preds = {}
            for outp in mp_state.output_preds:
                is_not_done = nets._is_not_done_broadcast(lengths, i,
                                                          output_preds_cand[outp])
                output_preds[outp] = is_not_done * output_preds_cand[outp] + (
                    1.0 - is_not_done) * mp_state.output_preds[outp]

        new_mp_state = _MessagePassingWithMsgScanState(
            hint_preds=hint_preds,
            output_preds=output_preds,
            hiddens=hiddens,
            lstm_state=lstm_state,
            msgs=msgs_state,
            input_msg=input_msg_state,
            input_algo=input_algo_state
            )
        # Save memory by not stacking unnecessary fields
        accum_mp_state = _MessagePassingWithMsgScanState(
            hint_preds=hint_preds if return_hints else None,
            output_preds=output_preds if return_all_outputs else None,
            hiddens=None, lstm_state=None,
            msgs=msgs_state, input_msg=input_msg_state, input_algo=input_algo_state)
        
        # ^ Note: could implement the following to save memory:
        # if repred:
        #     msgs = l1_norm(msgs_state)
        #     input_msg = None
        #     input_algo = None
        # else:
        #     msgs = msgs_state
        #     input_msg = input_msg_state
        #     input_algo = input_algo_state
        
        # Complying to jax.scan, the first returned value is the state we carry over
        # the second value is the output that will be stacked over steps.
        return new_mp_state, accum_mp_state

    def __call__(self, features_list: List[nets._Features], repred: bool,
                 algorithm_index: int,
                 return_hints: bool,
                 return_all_outputs: bool):
        """Process one batch of data.

        Args:
          features_list: A list of _Features objects, each with the inputs, hints
            and lengths for a batch o data corresponding to one algorithm.
            The list should have either length 1, at train/evaluation time,
            or length equal to the number of algorithms this Net is meant to
            process, at initialization.
          repred: False during training, when we have access to ground-truth hints.
            True in validation/test mode, when we have to use our own
            hint predictions.
          algorithm_index: Which algorithm is being processed. It can be -1 at
            initialisation (either because we are initialising the parameters of
            the module or because we are intialising the message-passing state),
            meaning that all algorithms should be processed, in which case
            `features_list` should have length equal to the number of specs of
            the Net. Otherwise, `algorithm_index` should be
            between 0 and `length(self.spec) - 1`, meaning only one of the
            algorithms will be processed, and `features_list` should have length 1.
          return_hints: Whether to accumulate and return the predicted hints,
            when they are decoded.
          return_all_outputs: Whether to return the full sequence of outputs, or
            just the last step's output.

        Returns:
          A 2-tuple with (output predictions, hint predictions)
          for the selected algorithm.
        """
        if algorithm_index == -1:
            algorithm_indices = range(len(features_list))
        else:
            algorithm_indices = [algorithm_index]
        assert len(algorithm_indices) == len(features_list)

        self.encoders, self.decoders = self._construct_encoders_decoders()
        self.processor = self.processor_factory(self.hidden_dim, self.msg_dim)

        # Optionally construct LSTM.
        if self.use_lstm:
            self.lstm = hk.LSTM(
                hidden_size=self.hidden_dim,
                name='processor_lstm')
            lstm_init = self.lstm.initial_state
        else:
            self.lstm = None
            lstm_init = lambda x: 0

        for algorithm_index, features in zip(algorithm_indices, features_list):
            inputs = features.inputs
            hints = features.hints
            lengths = features.lengths

            batch_size, nb_nodes = nets._data_dimensions(features)

            nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
            hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))

            if self.use_lstm:
                lstm_state = lstm_init(batch_size * nb_nodes)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [
                                                                    b, n, -1]),
                    lstm_state)
            else:
                lstm_state = None

            mp_state = _MessagePassingWithMsgScanState(
                hint_preds=None, output_preds=None,
                hiddens=hiddens, lstm_state=lstm_state,
                msgs=None, input_msg=None, input_algo=None)

            # Do the first step outside of the scan because it has a different
            # computation graph.
            common_args = dict(
                hints=hints,
                repred=repred,
                inputs=inputs,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                lengths=lengths,
                spec=self.spec[algorithm_index],
                encs=self.encoders[algorithm_index],
                decs=self.decoders[algorithm_index],
                return_hints=return_hints,
                return_all_outputs=return_all_outputs,
            )
            mp_state, lean_mp_state = self._msg_passing_step(
                mp_state,
                i=0,
                first_step=True,
                **common_args)

            # Then scan through the rest.
            scan_fn = functools.partial(
                self._msg_passing_step,
                first_step=False,
                **common_args)
            
            output_mp_state, accum_mp_state = hk.scan(
                scan_fn,
                mp_state,
                jnp.arange(nb_mp_steps - 1) + 1,
                length=nb_mp_steps - 1)

        # We only return the last algorithm's output. That's because
        # the output only matters when a single algorithm is processed; the case
        # `algorithm_index==-1` (meaning all algorithms should be processed)
        # is used only to init parameters.
        accum_mp_state = jax.tree_util.tree_map(
            lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
            lean_mp_state, accum_mp_state)
        
        def invert(d):
            """Dict of lists -> list of dicts."""
            if d:
                return [dict(zip(d, i)) for i in zip(*d.values())]

        if return_all_outputs:
            output_preds = {k: jnp.stack(v)
                            for k, v in accum_mp_state.output_preds.items()}
        else:
            output_preds = output_mp_state.output_preds
        hint_preds = invert(accum_mp_state.hint_preds)

        all_msgs = accum_mp_state.msgs
        # shape: (num_steps, layers_per_hint = 1, num_samples, num_nodes, num_nodes, msg_dim)
        all_input_msg = accum_mp_state.input_msg
        # shape: (num_steps, layers_per_hint = 1, num_samples, num_nodes, num_nodes, 4*hidden_dim)
        all_input_algo = accum_mp_state.input_algo
        # shape: (num_steps, layers_per_hint = 1, num_samples, num_nodes, num_nodes, num_input_feats)
        
        # note: the following only works when layers_per_hint = 1
        
        all_msgs = all_msgs.squeeze()
        all_input_msg = all_input_msg.squeeze()
        all_input_algo = all_input_algo.squeeze()
        # shape: (num_steps, num_samples, num_nodes, num_nodes, ^)
        
        all_msgs = jnp.transpose(all_msgs, (1, 0, 2, 3, 4))
        all_input_msg = jnp.transpose(all_input_msg, (1, 0, 2, 3, 4))
        all_input_algo = jnp.transpose(all_input_algo, (1, 0, 2, 3, 4))
        # shape: (num_samples, num_steps, num_nodes, num_nodes, ^)

        return output_preds, hint_preds, all_msgs, all_input_msg, all_input_algo


    def _one_step_pred(
        self,
        inputs: nets._Trajectory,
        hints: nets._Trajectory,
        hidden: nets._Array,
        batch_size: int,
        nb_nodes: int,
        lstm_state: Optional[hk.LSTMState],
        spec: nets._Spec,
        encs: Dict[str, List[hk.Module]],
        decs: Dict[str, Tuple[hk.Module]],
        repred: bool,
    ):
        """Generates one-step predictions."""

        # Initialise empty node/edge/graph features and adjacency matrix.
        node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
        edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
        graph_fts = jnp.zeros((batch_size, self.hidden_dim))
        adj_mat = jnp.repeat(
            jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        trajectories = [inputs]
        if self.encode_hints:
            trajectories.append(hints)
        
        def accum_input_algo(input_algo, new_input):
            new_input = jnp.expand_dims(new_input, axis=-1)
            if input_algo is None:
                return new_input
            else:
                return jnp.concatenate((input_algo, new_input), axis=-1)
        
        input_algo = None
        
        for trajectory in trajectories:
            for dp in trajectory:
                # try:
                #     dp = encoders.preprocess(dp, nb_nodes)
                #     assert dp.type_ != nets._Type.SOFT_POINTER
                #     adj_mat = encoders.accum_adj_mat(dp, adj_mat)
                #     encoder = encs[dp.name]
                #     edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
                #     node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
                #     graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)
                #         
                #     if dp.location == _Location.NODE:
                #         node_input_1 = jnp.tile(jnp.expand_dims(dp.data, axis=1), reps=(1, nb_nodes, 1, 1))
                #         node_input_2 = jnp.tile(jnp.expand_dims(dp.data, axis=2), reps=(1, 1, nb_nodes, 1))
                #         input_algo = accum_input_algo(input_algo, node_input_1)
                #         input_algo = accum_input_algo(input_algo, node_input_2)
                #     elif dp.location == _Location.EDGE:
                #         edge_input = dp.data
                #         input_algo = accum_input_algo(input_algo, edge_input)
                #     elif dp.location == _Location.GRAPH:
                #         graph_input = np.tile(jnp.expand_dims(dp.data, axis=(1, 2)), reps=(1, nb_nodes, nb_nodes, 1))
                #         input_algo = accum_input_algo(input_algo, graph_input)
                #         
                # except Exception as e:
                #     raise Exception(f'Failed to process {dp}') from e
                
                if dp.location == _Location.NODE:
                    node_input_1 = jnp.tile(jnp.expand_dims(dp.data, axis=1), reps=(1, nb_nodes, 1))
                    node_input_2 = jnp.tile(jnp.expand_dims(dp.data, axis=2), reps=(1, 1, nb_nodes))
                    input_algo = accum_input_algo(input_algo, node_input_1)
                    input_algo = accum_input_algo(input_algo, node_input_2)
                elif dp.location == _Location.EDGE:
                    edge_input = dp.data
                    input_algo = accum_input_algo(input_algo, edge_input)
                elif dp.location == _Location.GRAPH:
                    graph_input = jnp.tile(jnp.expand_dims(dp.data, axis=(1, 2)), reps=(1, nb_nodes, nb_nodes))
                    input_algo = accum_input_algo(input_algo, graph_input)
                
                dp = encoders.preprocess(dp, nb_nodes)
                assert dp.type_ != nets._Type.SOFT_POINTER
                adj_mat = encoders.accum_adj_mat(dp, adj_mat)
                encoder = encs[dp.name]
                edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
                node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
                graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)
                    
        # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        nxt_hidden = hidden
        msgs = None
        input_msg = None
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge, nxt_msg, nxt_input_msg = self.processor(
                node_fts,
                edge_fts,
                graph_fts,
                adj_mat,
                nxt_hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
            )
            if msgs is None:
                msgs = jnp.expand_dims(nxt_msg, axis=0)
                input_msg = jnp.expand_dims(nxt_input_msg, axis=0)
            else:
                msgs = jnp.concatenate([msgs, nxt_msg], axis=0)
                input_msg = jnp.concatenate([input_msg, next_input_msg], axis=0)
        
        if not repred:      # dropout only on training
            nxt_hidden = hk.dropout(
                hk.next_rng_key(), self._dropout_prob, nxt_hidden)

        if self.use_lstm:
            # lstm doesn't accept multiple batch dimensions (in our case, batch and
            # nodes), so we vmap over the (first) batch dimension.
            nxt_hidden, nxt_lstm_state = jax.vmap(
                self.lstm)(nxt_hidden, lstm_state)
        else:
            nxt_lstm_state = None

        h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)
        if nxt_edge is not None:
            e_t = jnp.concatenate([edge_fts, nxt_edge], axis=-1)
        else:
            e_t = edge_fts

        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.
        hint_preds, output_preds = decoders.decode_fts(
            decoders=decs,
            spec=spec,
            h_t=h_t,
            adj_mat=adj_mat,
            edge_fts=e_t,
            graph_fts=graph_fts,
            inf_bias=self.processor.inf_bias,
            inf_bias_edge=self.processor.inf_bias_edge,
            repred=repred,
        )

        return nxt_hidden, output_preds, hint_preds, nxt_lstm_state, msgs, input_msg, input_algo
