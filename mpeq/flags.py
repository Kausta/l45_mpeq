class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

FLAGS_dict = {
	'algorithms': ['binary_search'],
	'train_lengths': ['4', '7', '11', '13', '16'],
	'length_needle': -8,
	'seed': 42,
	'random_pos': True,
	'enforce_permutations': True,
	'enforce_pred_as_input': True,
	'batch_size': 32,
	'chunked_training': False,
	'chunk_length': 16,
	'train_steps': 400,
	'eval_every': 50,
	'test_every': 100,
	'hidden_size': 8,
	'msg_size': 64,
	'nb_heads': 1,
	'nb_msg_passing_steps': 1,
	'learning_rate': 0.001,
	'grad_clip_max_norm': 1.0,
	'dropout_prob': 0.0,
	'hint_teacher_forcing': 0.0,
	'l1_weight': 0.001,
	'hint_mode': 'encoded_decoded',
	'hint_repred_mode': 'soft',
	'use_ln': True,
	'use_lstm': False,
	'nb_triplet_fts': 8,
	'encoder_init': 'xavier_on_scalars',
	'processor_type': 'pgn_with_msg',
	'checkpoint_path': '/tmp/CLRS30',
	'dataset_path': '/tmp/CLRS30',
	'freeze_processor': False,
}

FLAGS = dotdict(FLAGS_dict)
