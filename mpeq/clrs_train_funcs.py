import functools
import os
import shutil
from typing import Any, Dict, List

import logging
import clrs
import jax
import jax.numpy as jnp
import numpy as np
import requests
import tensorflow as tf

import model
import flags
from clrs_train_funcs import *


FLAGS = flags.FLAGS

PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march',
]

def unpack(v):
    try:
        return v.item()  # DeviceArray
    except (AttributeError, ValueError):
        return v


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
    """Download CLRS30 dataset if needed."""
    dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
    if os.path.isdir(dataset_folder):
        logging.info('Dataset found at %s. Skipping download.', dataset_folder)
        return dataset_folder
    logging.info('Dataset not found in %s. Downloading...', dataset_folder)

    clrs_url = clrs.get_dataset_gcp_url()
    request = requests.get(clrs_url, allow_redirects=True)
    clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
    os.makedirs(dataset_folder)
    open(clrs_file, 'wb').write(request.content)
    shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
    os.remove(clrs_file)
    return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
    """Create a sampler with given options.

    Args:
      length: Size of samples (i.e., number of nodes in the graph).
        A length of -1 will mean that the benchmark
        dataset (for the given split) is used. Positive sizes will instantiate
        samplers of the corresponding size.
      rng: Numpy random state.
      algorithm: The name of the algorithm to sample from.
      split: 'train', 'val' or 'test'.
      batch_size: Samples per batch.
      multiplier: Integer multiplier for the number of samples in the dataset,
        only used for positive sizes. Negative multiplier means infinite samples.
      randomize_pos: Whether to randomize the `pos` input.
      enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
      enforce_permutations: Whether to enforce permutation pointers.
      chunked: Whether to chunk the dataset.
      chunk_length: Unroll length of chunks, if `chunked` is True.
      sampler_kwargs: Extra args passed to the sampler.
    Returns:
      A sampler (iterator), the number of samples in the iterator (negative
      if infinite samples), and the spec.
    """
    if length < 0:  # load from file
        dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
        sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                         algorithm=algorithm,
                                                         batch_size=batch_size,
                                                         split=split)
        sampler = sampler.as_numpy_iterator()
    else:
        num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
        sampler, spec = clrs.build_sampler(
            algorithm,
            seed=rng.randint(2**32),
            num_samples=num_samples,
            length=length,
            **sampler_kwargs,
        )
        sampler = _iterate_sampler(sampler, batch_size)

    if randomize_pos:
        sampler = clrs.process_random_pos(sampler, rng)
    if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
        spec, sampler = clrs.process_pred_as_input(spec, sampler)
    spec, sampler = clrs.process_permutations(
        spec, sampler, enforce_permutations)
    if chunked:
        sampler = clrs.chunkify(sampler, chunk_length)
    return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
    """Create a sampler with cycling sample sizes."""
    ss = []
    tot_samples = 0
    for length in sizes:
        sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
        ss.append(sampler)
        tot_samples += num_samples

    def cycle_samplers():
        while True:
            for s in ss:
                yield next(s)
    return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
    return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
    """Collect batches of output and hint preds and evaluate them."""
    processed_samples = 0
    preds = []
    outputs = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        new_rng_key, rng_key = jax.random.split(rng_key)
        # TODO: Can take messages here if required
        # see below
        cur_preds, _, _, _ = predict_fn(new_rng_key, feedback.features)
        preds.append(cur_preds)
        processed_samples += batch_size
    outputs = _concat(outputs, axis=0)
    preds = _concat(preds, axis=0)
    out = clrs.evaluate(outputs, preds)
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}


def get_msgs(sampler, predict_fn, sample_count, rng_key, sample_prob=0.001):
    """Get messages from model.
    
    CAUTION: size of msgs can get large very quickly, so beware when
        running with a large number of samples.
    Use sample_prob to reduce the number of messages that are saved
    by randomly sampling messages
    
    Note: we can only perform symbolic regression on a maximum of
    ~5000 messages anyways, so we can use sample_prob to get to this threshold
    """
    processed_samples = 0
    msgs = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        new_rng_key, rng_key = jax.random.split(rng_key)
        _, _, cur_msgs, cur_input_msg = predict_fn(new_rng_key, feedback.features)
        
        cur_msgs = cur_msgs.reshape(-1, cur_msgs.shape[-1])
        cur_input_msg = cur_input_msg.reshape(-1, cur_input_msg.shape[-1])
        cur_msg_concat = jnp.concatenate((cur_msgs, cur_input_msg), axis=-1)
        
        new_rng_key, rng_key = jax.random.split(rng_key)
        mask = jax.random.choice(new_rng_key,
                                 a=jnp.array([False, True]),
                                 shape=(cur_msg_concat.shape[0],),
                                 p=jnp.array([1 - sample_prob, sample_prob]),
                                 replace=True,)
        cur_msg_concat = cur_msg_concat[mask]
        
        msgs.append(cur_msg_concat)
        processed_samples += batch_size
    msgs = _concat(msgs, axis=0)
    
    return msgs


def create_samplers(rng, train_lengths: List[int]):
    """Create all the samplers."""
    train_samplers = []
    val_samplers = []
    val_sample_counts = []
    test_samplers = []
    test_sample_counts = []
    spec_list = []

    for algo_idx, algorithm in enumerate(FLAGS.algorithms):
        # Make full dataset pipeline run on CPU (including prefetching).
        with tf.device('/cpu:0'):

            if algorithm in ['naive_string_matcher', 'kmp_matcher']:
                # Fixed haystack + needle; variability will be in needle
                # Still, for chunked training, we maintain as many samplers
                # as train lengths, since, for each length there is a separate state,
                # and we must keep the 1:1 relationship between states and samplers.
                max_length = max(train_lengths)
                if max_length > 0:  # if < 0, we are using the benchmark data
                    max_length = (max_length * 5) // 4
                train_lengths = [max_length]
                if FLAGS.chunked_training:
                    train_lengths = train_lengths * len(train_lengths)

            logging.info('Creating samplers for algo %s', algorithm)

            p = tuple([0.1 + 0.1 * i for i in range(9)])
            if p and algorithm in ['articulation_points', 'bridges',
                                   'mst_kruskal', 'bipartite_matching']:
                # Choose a lower connection probability for the above algorithms,
                # otherwise trajectories are very long
                p = tuple(np.array(p) / 2)
            length_needle = FLAGS.length_needle
            sampler_kwargs = dict(p=p, length_needle=length_needle)
            if length_needle == 0:
                sampler_kwargs.pop('length_needle')

            common_sampler_args = dict(
                algorithm=FLAGS.algorithms[algo_idx],
                rng=rng,
                enforce_pred_as_input=FLAGS.enforce_pred_as_input,
                enforce_permutations=FLAGS.enforce_permutations,
                chunk_length=FLAGS.chunk_length,
            )

            train_args = dict(sizes=train_lengths,
                              split='train',
                              batch_size=FLAGS.batch_size,
                              multiplier=-1,
                              randomize_pos=FLAGS.random_pos,
                              chunked=FLAGS.chunked_training,
                              sampler_kwargs=sampler_kwargs,
                              **common_sampler_args)
            train_sampler, _, spec = make_multi_sampler(**train_args)

            mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
            val_args = dict(sizes=[np.amax(train_lengths)],
                            split='val',
                            batch_size=32,
                            multiplier=2 * mult,
                            randomize_pos=FLAGS.random_pos,
                            chunked=False,
                            sampler_kwargs=sampler_kwargs,
                            **common_sampler_args)
            val_sampler, val_samples, spec = make_multi_sampler(**val_args)

            test_args = dict(sizes=[-1],
                             split='test',
                             batch_size=32,
                             multiplier=2 * mult,
                             randomize_pos=False,
                             chunked=False,
                             sampler_kwargs={},
                             **common_sampler_args)
            test_sampler, test_samples, spec = make_multi_sampler(**test_args)

        spec_list.append(spec)
        train_samplers.append(train_sampler)
        val_samplers.append(val_sampler)
        val_sample_counts.append(val_samples)
        test_samplers.append(test_sampler)
        test_sample_counts.append(test_samples)

    return (train_samplers,
            val_samplers, val_sample_counts,
            test_samplers, test_sample_counts,
            spec_list)
