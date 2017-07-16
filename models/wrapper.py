# should contain wrapper classes
from hred.dialog_encdec import DialogEncoderDecoder
from hred.state import prototype_state
import cPickle
import hred.search as search


class HRED_Wrapper(object):

    def __init__(self, model_prefix, dict_file, name):
        # Load the HRED model.
        self.name = name
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

        self.speaker_token = '<first_speaker>'
        if name == 'reddit':
            self.speaker_token = '<speaker_1>'

        self.remove_tokens = ['<first_speaker>', '<at>', '<second_speaker>']
        for i in range(0, 10):
            self.remove_tokens.append('<speaker_%d>' % i)

    def _preprocess(self, text):
        text = text.replace("'", " '")
        text = '%s %s </s>' % (self.speaker_token, text.strip().lower())
        return text

    def _format_output(self, text):
        text = text.replace(" '", "'")
        for token in self.remove_tokens:
            text = text.replace(token, '')
        return text

    # must contain this method for the bot
    def get_response(self, user_id, text, context):
        print '--------------------------------'
        print 'Generating HRED response for user %s.' % user_id
        text = self._preprocess(text)
        context.append(text)
        context = list(context)
        print 'Using context: %s' % ' '.join(context)
        samples, costs = self.sampler.sample([' '.join(context),], ignore_unk=True, verbose=False, return_words=True)
        response = samples[0][0].replace('@@ ', '').replace('@@', '')
        context.append(response)
        response = self._format_output(response)
        print 'Response: %s' % response
        return response, context

