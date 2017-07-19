# should contain wrapper classes

import logging
import cPickle
import numpy as np
import theano
import re
import lasagne

from hred.dialog_encdec import DialogEncoderDecoder
from hred.state import prototype_state
import hred.search as search

from dual_encoder.model import Model as DE_Model
from hredqa.hred_pytorch import HRED_QA

logger = logging.getLogger(__name__)


class Model_Wrapper(object):
    """
    Super class for all Model wrappers
    """

    def __init__(self, model_prefix, name):
        """
        Default constructor
        :param model_prefix: path to the model files
        :param name: model name
        """
        self.model_prefix = model_prefix
        self.name = name

    def get_response(self, user_id, text, context):
        """
        Generate a new response, and add it to the context
        :param user_id: id of the person we chat with
        :param text: the new utterance we just received
        :type text: str
        :param context: queue of conversation history: sliding window of most recent utterances
        :type context: collections.dequeue
        :return: the generated response as well as the new context
        """
        pass  # TO BE IMPLEMENTED IN SUB-CLASSES


class HRED_Wrapper(Model_Wrapper):

    def __init__(self, model_prefix, dict_file, name):
        # Load the HRED model.
        super(HRED_Wrapper, self).__init__(model_prefix, name)
        state_path = '%s_state.pkl' % model_prefix
        model_path = '%s_model.npz' % model_prefix

        state = prototype_state()
        with open(state_path, 'r') as handle:
            state.update(cPickle.load(handle))
        state['dictionary'] = dict_file
        print 'Building %s model...' % name
        self.model = DialogEncoderDecoder(state)
        print 'Building sampler...'
        self.sampler = search.BeamSampler(self.model)
        print 'Loading model...'
        self.model.load(model_path)
        print 'Model built (%s).' % name

        self.speaker_token = ['<first_speaker>', '<second_speaker>']
        if name == 'hred-reddit':
            self.speaker_token = ['<speaker_1>', '<speaker_2']

    def _preprocess(self, text, context_length):
        text = text.replace("'", " '")  # TODO: apply same tokenization script as in data creating
        text = '%s %s </s>' % (self.speaker_token[context_length%2], text.strip().lower())
        return text

    def _format_output(self, text):
        text = text.replace(" '", "'")
        text = re.sub('<[^>]+>', '', text) # remove all <tags>
        return ' '.join(text.strip().split())  # strip, split, join to remove extra spaces

    # must contain this method for the bot
    def get_response(self, user_id, text, context):
        logger.info('--------------------------------')
        logger.info('Generating HRED response for user %s.' % user_id)
        text = self._preprocess(text, len(context))
        context.append(text)
        logger.info('Using context: %s' % ' '.join(list(context)))
        samples, costs = self.sampler.sample([' '.join(list(context)),], ignore_unk=True, verbose=False, return_words=True)
        response = samples[0][0].replace('@@ ', '').replace('@@', '')
        context.append(response)
        response = self._format_output(response)
        logger.info('Response: %s' % response)
        return response, context


class Dual_Encoder_Wrapper(Model_Wrapper):

    def __init__(self, model_prefix, data_fname, dict_fname, name):
        super(Dual_Encoder_Wrapper, self).__init__(model_prefix, name)

        try:
            with open('%s_model.pkl' % model_prefix, 'rb') as handle:
                self.model = cPickle.load(handle)
        except Exception as e:
            logger.error("%s\n ERROR: couldn't load the model" % e)
            logger.info("Will create a new one with pretrained parameters")
            # Loading old arguments
            with open('%s_args.pkl' % model_prefix, 'rb') as handle:
                old_args = cPickle.load(handle)

            logger.info("Loading data...")
            with open('%s' % data_fname, 'rb') as handle:
                train_data, val_data, test_data = cPickle.load(handle)
            data = {'train': train_data, 'val': val_data, 'test': test_data}
            # W is the word embedding matrix and word2idx, idx2word are dictionaries
            with open('%s' % dict_fname, 'rb') as handle:
                word2idx, idx2word = cPickle.load(handle)
            W = np.zeros(shape=(len(word2idx), old_args.emb_size))
            for idx in idx2word:
                W[idx] = np.random.uniform(-0.25, 0.25, old_args.emb_size)
            logger.info("W.shape: %s" % (W.shape,))

            logger.info("Creating model...")
            self.model = self._create_model(data, W, word2idx, idx2word, old_args)

            logger.info("Set the learned weights...")
            with open('%s_best_weights.pkl' % model_prefix, 'rb') as handle:
                params = cPickle.load(handle)
                lasagne.layers.set_all_param_values(self.model.l_out, params)
            with open('%s_best_M.pkl' % model_prefix, 'rb') as handle:
                M = cPickle.load(handle)
                self.model.M.set_value(M)
            with open('%s_best_embed.pkl' % model_prefix, 'rb') as handle:
                em = cPickle.load(handle)
                self.model.embeddings.set_value(em)

        with open('%s_timings.pkl' % model_prefix, 'rb') as handle:
            timings = cPickle.load(handle)
            self.model.timings = timings  # load last timings (when no improvement was done)
        logger.info("Model loaded.")

        with open("%s_r-encs.pkl" % model_prefix, 'rb') as handle:
            self.cached_retrieved_data = cPickle.load(handle)

        self.speaker_tokens = ['<first_speaker>', '<second_speaker>']

    def _create_model(self, data, w, word2idx, idx2word, args):
        return DE_Model(
            data=data,
            W=w.astype(theano.config.floatX),
            word2idx=word2idx,
            idx2word=idx2word,
            save_path=args.save_path,
            save_prefix=args.save_prefix,
            max_seqlen=args.max_seqlen,  # default 160
            batch_size=args.batch_size,  # default 256
            # Network architecture:
            encoder=args.encoder,  # default RNN
            hidden_size=args.hidden_size,  # default 200
            n_recurrent_layers=args.n_recurrent_layers,  # default 1
            is_bidirectional=args.is_bidirectional,  # default False
            dropout_out=args.dropout_out,  # default 0.
            dropout_in=args.dropout_in,  # default 0.
            # Learning parameters:
            patience=args.patience,  # default 10
            optimizer=args.optimizer,  # default ADAM
            lr=args.lr,  # default 0.001
            lr_decay=args.lr_decay,  # default 0.95
            fine_tune_W=args.fine_tune_W,  # default False
            fine_tune_M=args.fine_tune_M,  # default False
            # NTN parameters:
            use_ntn=args.use_ntn,  # default False
            k=args.k,  # default 4
            # Regularization parameters:
            penalize_emb_norm=args.penalize_emb_norm,  # default False
            penalize_emb_drift=args.penalize_emb_drift,  # default False
            emb_penalty=args.emb_penalty,  # default 0.001
            penalize_activations=args.penalize_activations,  # default False
            act_penalty=args.act_penalty  # default 500
        )

    def _preprocess(self, text, context_length):
        text = text.replace("'", " '")  # TODO: apply same tokenization script as in data creating
        text = '%s %s </s>' % (self.speaker_tokens[context_length % 2], text.strip().lower())
        return text

    def _format_output(self, text):
        text = text.replace(" '", "'")  # TODO: come up with a smarter reverse tokenization system?
        text = re.sub('<[^>]+>', '', text) # remove all <tags>
        return ' '.join(text.strip().split())  # strip, split, join to remove extra spaces

    def get_response(self, user_id, text, context):
        logger.info('--------------------------------')
        logger.info('Generating DE response for user %s.' % user_id)
        text = self._preprocess(text, len(context))
        context.append(text)
        logger.info('Using context: %s' % ' '.join(list(context)))

        cached_retrieved_data = self.model.retrieve(context_set=[' '.join(context)],
                                                   response_set=self.cached_retrieved_data['r'],
                                                   response_embs=self.cached_retrieved_data['r_embs'],
                                                   k=1, batch_size=1, verbose=False)
        response = cached_retrieved_data['r_retrieved'][0][0].replace('@@ ', '').replace('@@', '')

        response = self._format_output(response)
        context.append(response) 
        logger.info('Response: %s' % response)
        return response, context

class HREDQA_Wrapper(Model_Wrapper):
    def __init__(self, model_prefix, dict_fname, name):
        super(HREDQA_Wrapper, self).__init__(model_prefix, name)

        self.model = HRED_QA(dictionary=dict_fname,
                encoder_file='{}encoder_5.model'.format(model_prefix),
                decoder_file='{}decoder_5.model'.format(model_prefix),
                context_file='{}context_5.model'.format(model_prefix)
                )

	self.speaker_tokens = ['<first_speaker>', '<second_speaker>']
 
    def _preprocess(self, text, context_length):
        text = text.replace("'", " '")  # TODO: apply same tokenization script as in data creating
        text = '%s %s </s>' % (self.speaker_tokens[context_length % 2], text.strip().lower())
        return text

    def _get_sentences(self,context):
        sents = [re.sub('<[^>]+>', '', p) for p in context]
        return sents

    def _format_output(self, text):
        text = text.replace(" '", "'")  # TODO: come up with a smarter reverse tokenization system?
        text = re.sub('<[^>]+>', '', text)
        if '?' not in text:
            text = text + ' ?'
        return ' '.join(text.strip().split())  # strip, split, join to remove extra spaces

    def get_response(self, user_id, text, context):
        logger.info('------------------------------------')
        logger.info('Generating Followup question for user %s.' % user_id)
        text = self._preprocess(text, len(context))
        context.append(text)
        logger.info('Using context: %s' % ' '.join(list(context)))

        response = self.model.evaluate(self.model.encoder_model,self.model.decoder_model,self.model.context_model,self._get_sentences(context))
        response = ' '.join(response)
        response = self._format_output(response)
        context.append(self._preprocess(response,len(context)))
        return response,context



