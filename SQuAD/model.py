import tensorflow as tf
import cPickle
import numpy as np
np.random.seed(10)

N_HIDDEN = 100
MAX_LENGTH = 100
BATCH_SIZE = 32
WE_DIM = 100
DATA_FILENAME = '/home/ml/mnosew1/data/squad/SQuAD_Level2_Dataset.pkl'
EMB_FILENAME = '/home/ml/mnosew1/data/squad/SQuAD_glove_embedding.pkl'
VAL_FREQUENCY = 1000
TRAINING_ITERS = 10000
DISPLAY_FREQ = 50
LOG_FREQ = 10

def SpanModel(x_q, x_a, seqlen_q, seqlen_a, mask_a, y_span, embeddings):
	embs = tf.Variable(embeddings, name='word_embeddings', dtype=tf.float32)

	x_q_emb = tf.nn.embedding_lookup(embs, x_q)
	x_a_emb = tf.nn.embedding_lookup(embs, x_a)

	q_lstm = tf.contrib.rnn.LSTMCell(N_HIDDEN)

	f_lstm = tf.contrib.rnn.LSTMCell(N_HIDDEN)
	r_lstm = tf.contrib.rnn.LSTMCell(N_HIDDEN)

	s_lstm = tf.contrib.rnn.LSTMCell(N_HIDDEN)
	# Input the seqlen variable.
	outputs_a, _ = tf.nn.bidirectional_dynamic_rnn(f_lstm, r_lstm, x_a_emb, sequence_length=seqlen_a, dtype=tf.float32, scope='answer')
	outputs_a = tf.concat(outputs_a, 2)
	_, state_q = tf.nn.dynamic_rnn(q_lstm, x_q_emb, sequence_length=seqlen_q, dtype=tf.float32, scope='question')
	# Tile the question state.
	output_q = tf.reshape(tf.tile(state_q[0], [1, MAX_LENGTH]), [BATCH_SIZE, MAX_LENGTH, N_HIDDEN])
	# Concatenate the outputs with the state from the question.
	outputs = tf.concat([outputs_a, output_q], axis=2)

	# Have an LSTM working on this layer.
	outputs_span, _ = tf.nn.dynamic_rnn(s_lstm, outputs, sequence_length=seqlen_a, dtype=tf.float32, scope='span')

	# TODO: Have a fully connected layer to predict 0 or 1 for the span.
	W, b = tf.Variable(tf.random_normal([N_HIDDEN, 2]), name='W'), tf.Variable(tf.random_normal([2]), name='b')
	outputs_preds = tf.tensordot(outputs_span, W, axes=[[2],[0]]) + b

	return outputs_preds
	
def get_batch(data, indices, ix):
	# Prepate the batch.
	batch_ix = indices[ix*BATCH_SIZE:(ix+1)*BATCH_SIZE]

	x_q = np.zeros((BATCH_SIZE, MAX_LENGTH), dtype='int32')
	x_a = np.zeros((BATCH_SIZE, MAX_LENGTH), dtype='int32')
	seqlen_q = np.zeros((BATCH_SIZE,), dtype='int32')
	seqlen_a = np.zeros((BATCH_SIZE,), dtype='int32')
	mask_a = np.zeros((BATCH_SIZE, MAX_LENGTH), dtype='float32')
	y_span = np.zeros((BATCH_SIZE, MAX_LENGTH), dtype='int32')

	for b_ix, d_ix in enumerate(batch_ix):
		question = data['c'][d_ix][:MAX_LENGTH]
		answer = data['r'][d_ix][:MAX_LENGTH]
		span = data['am'][d_ix][:MAX_LENGTH]
		x_q[b_ix, :len(question)] = question
		x_a[b_ix, :len(answer)] = answer
		seqlen_q[b_ix] = len(question)
		seqlen_a[b_ix] = len(answer)
		y_span[b_ix, :len(span)] = span
		mask_a[b_ix, :len(answer)] = 1

	return {'x_q': x_q, 'x_a': x_a, 
			'seqlen_q': seqlen_q, 'seqlen_a': seqlen_a, 
			'y_span': y_span, 'mask_a': mask_a}

if __name__ == '__main__':

	print 'Loading data...'
	with open(DATA_FILENAME, 'rb') as handle:
		train_data, val_data = cPickle.load(handle)
	with open(EMB_FILENAME, 'rb') as handle:
		embeddings = cPickle.load(handle)

	indices = np.arange(len(train_data['c']))
	n_train_batches = len(indices) // BATCH_SIZE
	n_val_batches = len(val_data['c']) // BATCH_SIZE

	print 'Building the model...'
	# Define the inputs to the model.
	x_q = tf.placeholder(tf.int32, shape=(BATCH_SIZE, MAX_LENGTH), name='q_ix')
	x_a = tf.placeholder(tf.int32, shape=(BATCH_SIZE, MAX_LENGTH), name='a_ix')
	seqlen_q = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='q_seqlen')
	seqlen_a = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='a_seqlen')
	mask_a = tf.placeholder(tf.float32, shape=(BATCH_SIZE, MAX_LENGTH), name='a_mask')
	y_span = tf.placeholder(tf.int32, shape=(BATCH_SIZE, MAX_LENGTH), name='y_span')

	# Build the model.
	pred = SpanModel(x_q, x_a, seqlen_q, seqlen_a, mask_a, y_span, embeddings)

	# Build the training function.
	cost = tf.contrib.seq2seq.sequence_loss(pred, y_span, weights=mask_a)
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	tf.summary.scalar('cost', cost)

	# Write accuracy/exact-matches code.
	hard_preds = tf.cast(tf.argmax(pred, axis=2), tf.int32, name='Predictions')
	with tf.variable_scope('Accuracy'):
		acc = tf.reduce_sum(tf.cast(tf.equal(hard_preds,y_span), tf.float32)*mask_a)/tf.reduce_sum(mask_a)
		tf.summary.scalar('accuracy', acc)
	with tf.variable_scope('ExactMatches'):	
		incorrect = tf.cast(tf.not_equal(hard_preds, y_span), tf.float32)*mask_a
		errors_per_example = tf.reduce_any(tf.cast(incorrect, tf.bool), axis=1)
		ems = BATCH_SIZE - tf.reduce_sum(tf.cast(errors_per_example, tf.float32))
		tf.summary.scalar('exact_matches', ems)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('./Logs/train')
	val_writer = tf.summary.FileWriter('./Logs/val')

	print 'Starting training...'
	saver = tf.train.Saver(max_to_keep=5)
	init = tf.global_variables_initializer()
	# Start the training loop.
	with tf.Session() as sess:
		sess.run(init)
		step = 1

		train_writer.add_graph(sess.graph)

		best_val_loss = np.inf
		while step*BATCH_SIZE < TRAINING_ITERS:
			np.random.shuffle(indices)
			for ix in range(0, n_train_batches):
				batch_data = get_batch(train_data, indices, ix)
				summary, _, train_cost, accuracy, exact_matches = sess.run([merged, optimizer, cost, acc, ems], 
							{x_q: batch_data['x_q'],
									x_a: batch_data['x_a'],
									seqlen_q: batch_data['seqlen_q'],
									seqlen_a: batch_data['seqlen_a'],
									y_span: batch_data['y_span'],
									mask_a: batch_data['mask_a']
							})

				if step % DISPLAY_FREQ == 0:
					print 'Training cost: %f\tTraining Accuracy: %f\tTraining EMs:%f' % (train_cost, accuracy, exact_matches/float(BATCH_SIZE))
				if step % LOG_FREQ == 0:
					train_writer.add_summary(summary, step)
				step += 1

				# Perform Validation.
				if step % VAL_FREQUENCY == 0:

					val_losses, val_accs, val_ems = [], [], []
					for jx in range(0, n_val_batches):
						batch_data = get_batch(val_data, np.arange(0, len(val_data['c'])), jx)
						summary, val_cost, accuracy, exact_matches = sess.run([merged, cost, acc, ems], 
								{x_q: batch_data['x_q'],
									x_a: batch_data['x_a'],
									seqlen_q: batch_data['seqlen_q'],
									seqlen_a: batch_data['seqlen_a'],
									y_span: batch_data['y_span'],
									mask_a: batch_data['mask_a']
								})
						val_losses.append(val_cost)
						val_accs.append(accuracy)
						val_ems.append(exact_matches)
						val_writer.add_summary(summary, step)

					print 'Validation Accuracy: %f\tValidation EMs: %f' % (np.mean(val_accs), np.sum(val_ems)/(len(val_ems)*BATCH_SIZE))
					if np.mean(val_losses) < best_val_loss:
						best_val_loss = np.mean(val_losses)
						path = saver.save(sess, './Models')
						print 'Model improved. Save to %s' % path








