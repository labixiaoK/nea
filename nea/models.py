import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_model(args, initial_mean_value, overal_maxlen, vocab):
	
	import keras.backend as K
	from keras.layers.embeddings import Embedding
	from keras.models import Sequential, Model
	from keras.layers.core import Dense, Dropout, Activation
	from nea.my_layers import Attention, MeanOverTime, Conv1DWithMasking
	
	###############################################################################################################################
	## Recurrence unit type
	#

	if args.recurrent_unit == 'lstm':
		from keras.layers.recurrent import LSTM as RNN
	elif args.recurrent_unit == 'gru':
		from keras.layers.recurrent import GRU as RNN
	elif args.recurrent_unit == 'simple':
		from keras.layers.recurrent import SimpleRNN as RNN

	###############################################################################################################################
	## Create Model
	#
	
	dropout_W = 0.5		# default=0.5
	dropout_U = 0.1		# default=0.1
	cnn_border_mode='same'
	if initial_mean_value.ndim == 0:	#expand the dims
		initial_mean_value = np.expand_dims(initial_mean_value, axis=1)
	num_outputs = len(initial_mean_value)	#预测的分数种类数
	
	if args.model_type == 'cls':
		raise NotImplementedError
	
	#embedding-->cnn-->rnn(return_sequence=false)-->dropout-->dense-->sigmoid
	elif args.model_type == 'reg':
		logger.info('Building a REGRESSION model')
		model = Sequential()
		#确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。
		#如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 1
		#此处，input层省略是因为input_length有默认值
		model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))
		if args.cnn_dim > 0:	#border_mode==padding?? subsample_length==pooling?? where is the activation??
			model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
		if args.rnn_dim > 0:	#return_sequence 只返回最后一个 state
			model.add(RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U))
		if args.dropout_prob > 0:
			model.add(Dropout(args.dropout_prob))
		model.add(Dense(num_outputs))
		if not args.skip_init_bias:     #初始化最后一层layer的bias
			bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
			model.layers[-1].b.set_value(bias_value)
		model.add(Activation('sigmoid'))	#输出区间为（0，1）
		#设置model的embed层的序号，方便后续用预训练词向量的初始化，model的所有层都存在  model.layers 里
		model.emb_index = 0
	
	#embedding-->cnn-->rnn(return_sequence=true)-->dropout-->MeanoverTime or Attention(mean or sum)-->Dense-->sigmoid
	elif args.model_type == 'regp':
		logger.info('Building a REGRESSION model with POOLING')
		model = Sequential()
		model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))
		if args.cnn_dim > 0:
			model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
		if args.rnn_dim > 0:
			model.add(RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U))
		if args.dropout_prob > 0:
			model.add(Dropout(args.dropout_prob))
		if args.aggregation == 'mot':
			model.add(MeanOverTime(mask_zero=True))
		elif args.aggregation.startswith('att'):
			model.add(Attention(op=args.aggregation, activation='tanh', init_stdev=0.01))
		model.add(Dense(num_outputs))
		if not args.skip_init_bias:
			bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
			model.layers[-1].b.set_value(bias_value)
		model.add(Activation('sigmoid'))
		model.emb_index = 0
	#embedding-->cnn-->birnn(return_sequence=false)-->dropout-->merge(concat the forRnn&backRnn)-->dense-->sigmoid
	elif args.model_type == 'breg':
		logger.info('Building a BIDIRECTIONAL REGRESSION model')
		from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
		model = Sequential()	#这句应该是多余的
		sequence = Input(shape=(overal_maxlen,), dtype='int32')
		output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
		if args.cnn_dim > 0:
			output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
		if args.rnn_dim > 0:
			forwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U)(output)
			backwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
		if args.dropout_prob > 0:
			forwards = Dropout(args.dropout_prob)(forwards)
			backwards = Dropout(args.dropout_prob)(backwards)
		merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
		densed = Dense(num_outputs)(merged)
		if not args.skip_init_bias:
			raise NotImplementedError
		score = Activation('sigmoid')(densed)
		model = Model(input=sequence, output=score)
		model.emb_index = 1
	#embedding-->cnn-->biRnn(return_sequence=true)-->dropout-->meanOverTime-->merge(concat)-->dense-->sigmoid
	elif args.model_type == 'bregp':
		logger.info('Building a BIDIRECTIONAL REGRESSION model with POOLING')
		from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
		model = Sequential()  #多余的
		sequence = Input(shape=(overal_maxlen,), dtype='int32')
		output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
		if args.cnn_dim > 0:
			output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
		if args.rnn_dim > 0:
			forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(output)
			backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
		if args.dropout_prob > 0:
			forwards = Dropout(args.dropout_prob)(forwards)
			backwards = Dropout(args.dropout_prob)(backwards)
		forwards_mean = MeanOverTime(mask_zero=True)(forwards)
		backwards_mean = MeanOverTime(mask_zero=True)(backwards)
		merged = merge([forwards_mean, backwards_mean], mode='concat', concat_axis=-1)
		densed = Dense(num_outputs)(merged)
		if not args.skip_init_bias:
			raise NotImplementedError
		score = Activation('sigmoid')(densed)
		model = Model(input=sequence, output=score)
		model.emb_index = 1
	
	logger.info('  Done')
	
	###############################################################################################################################
	## Initialize embeddings if requested
	#

	if args.emb_path:
		from w2vEmbReader import W2VEmbReader as EmbReader
		logger.info('Initializing lookup table')
		emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
		model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
		logger.info('  Done')
	
	return model
