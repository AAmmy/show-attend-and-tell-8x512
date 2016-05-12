import numpy
from scipy.io import loadmat
import cPickle as pkl
from capgen import build_sampler, load_params, init_params, init_tparams, get_dataset
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import copy

# generate sample
def my_gen_sample_nonbeam(f_init, f_next, ctx0, options, maxlen=30, stochastic=False):
    # ctx0 = cc0[0]
    sample, sample_score = [], []
    live_k, dead_k = 1, 0
    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states, hyp_memories = [], []

    # only matters if we use lstm encoder
    # rval = [ctx, init_state, init_memory]
    # f_init = theano.function([ctx], [ctx]+init_state+init_memory, name='f_init', profile=False)
    rval = f_init(ctx0) # (ctx0 == rval[0]).all() # => True, len(rval[1])=len(rval[2])=1000

    ctx0 = rval[0] # 同じなので意味ない?
    next_state, next_memory = [], []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)  
    next_w = -1 * numpy.ones((1,)).astype('int64')
    w_i, w_p = [], []
    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx0]+next_state+next_memory))
        # f_next = theano.function([x, ctx]+init_state+init_memory, [next_probs, next_sample]+next_state+next_memory, name='f_next', profile=False)
        next_p = rval[0] # next_probs
        # next_w = rval[1] # next_sample, stochastic
        next_w = numpy.array([numpy.argmax(next_p)]) # next_sample, not stochastic
        w_i.append(next_w[0])
        w_p.append(next_p[0, next_w][0])
        if next_w[0] == 0: break
        # extract all the states and memories
        next_state, next_memory = [], []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])

    # w_w = " ".join(seq2words(w_i, word_idict))
    w_log_p = -numpy.log(w_p)
    w_score = w_log_p.sum()

    return w_i, w_log_p

# generate sample
def my_gen_sample(f_init, f_next, ctx0, options, stochastic=False, w_idx=[4, 12, 5, 2, 27, 100, 8, 80, 2, 83, 3, 0]):
    # ctx0 = cc0[0]
    sample, sample_score = [], []
    live_k, dead_k = 1, 0
    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states, hyp_memories = [], []

    # only matters if we use lstm encoder
    # rval = [ctx, init_state, init_memory]
    # f_init = theano.function([ctx], [ctx]+init_state+init_memory, name='f_init', profile=False)
    rval = f_init(ctx0) # (ctx0 == rval[0]).all() # => True, len(rval[1])=len(rval[2])=1000

    ctx0 = rval[0] # 同じなので意味ない?
    next_state, next_memory = [], []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)  
    next_w = -1 * numpy.ones((1,)).astype('int64')
    w_i, w_p = [], []
    for ii in w_idx:
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx0]+next_state+next_memory))
        # f_next = theano.function([x, ctx]+init_state+init_memory, [next_probs, next_sample]+next_state+next_memory, name='f_next', profile=False)
        next_p = rval[0] # next_probs
        # next_w = rval[1] # next_sample, stochastic
        next_w = numpy.array([ii]) # next_sample, not stochastic
        w_i.append(next_w[0])
        w_p.append(next_p[0, next_w][0])
        if next_w[0] == 0: break
        # extract all the states and memories
        next_state, next_memory = [], []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])

    # w_w = " ".join(seq2words(w_i, word_idict))
    w_log_p = -numpy.log(w_p)
    w_score = w_log_p.sum()

    return w_i, w_log_p

def gen_sample(f_init, f_next, ctx0, options, k=1, maxlen=30, stochastic=False):
    """
    f_init : theano function
        input: annotation, output: initial lstm state and memory 
        (also performs transformation on ctx0 if using lstm_encoder)
    f_next: theano function
        takes the previous word/state/memory + ctx0 and runs one
        step through the lstm
    Returns
    sample : list of list
        each sublist contains an (encoded) sample from the model 
    sample_score : numpy array
        scores of each sample
    """
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample, sample_score = [], []
    if stochastic:
        sample_score = 0

    live_k, dead_k = 1, 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states, hyp_memories = [], []

    # only matters if we use lstm encoder
    rval = f_init(ctx0)
    #ctx0 = rval[0]
    next_state, next_memory = [], []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)  
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx0]+next_state+next_memory))
        next_p = rval[0]
        next_w = rval[1]

        # extract all the states and memories
        next_state, next_memory = [], []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])

        if stochastic:
            sample.append(next_w[0]) # if we are using stochastic sampling this easy
            sample_score += next_p[0,next_w[0]]
            if next_w[0] == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p) 
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)] # (k-dead_k) numpy array of with min nll

            voc_size = next_p.shape[1]
            # indexing into the correct selected captions
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat] # extract costs from top hypothesis

            # a bunch of lists to hold future hypothesis
            new_hyp_samples, new_hyp_states = [], []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_states.append([])
            new_hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_memories.append([])

            # get the corresponding hypothesis and append the predicted word
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis 
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

            # check the finished samples for <eos> character
            new_live_k = 0
            hyp_samples, hyp_scores, hyp_states = [], [], []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_states.append([])
            hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_memories.append([])

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1 # completed sample!
                else:
                    new_live_k += 1 # collect collect correct states/memories
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_states[lidx].append(new_hyp_states[lidx][idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1 or dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = []
            for lidx in xrange(options['n_layers_lstm']):
                next_state.append(numpy.array(hyp_states[lidx]))
            next_memory = []
            for lidx in xrange(options['n_layers_lstm']):
                next_memory.append(numpy.array(hyp_memories[lidx]))

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score

def gen_model(model, options, sampling):
    trng = RandomStreams(1234)
    # this is zero indicate we are not using dropout in the graph
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    # get the parameters
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    # build the sampling computational graph
    # see capgen.py for more detailed explanations
    f_init, f_next = build_sampler(tparams, options, use_noise, trng, sampling=sampling)
    
    return f_init, f_next

def get_word_idict(worddict):
    word_idict = {v:k for k, v in worddict.iteritems()}
    word_idict[0], word_idict[1] = '<eos>', 'UNK'
    return word_idict

# index -> words
def seq2words(sample, word_idict):
    cap = []
    for w in sample:
        if w == 0:
            break
        cap.append(word_idict[w])
    return cap

def pad_y(y, zero_pad):
    res = []
    for y_ in y:
        cc = y_
        if zero_pad:
            cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
            cc0[:-1,:] = cc
        else:
            cc0 = cc
        res.append(cc0)
    return res

def get_sample_feats():
    feat_path = '../flickr8k/196x512_perfile_dense_name_resizecrop/'
    sample_feats = ['3110649716_c17e14670e.jpg', '543007912_23fc735b99.jpg',
                    '2813033949_e19fa08805.jpg', '247704641_d883902277.jpg',
                    '2398605966_1d0c9e6a20.jpg', '3718964174_cb2dc1615e.jpg',
                    '2522297487_57edf117f7.jpg', '2490768374_45d94fc658.jpg',
                    '2101457132_69c950bc45.jpg', '2909875716_25c8652614.jpg']
    y = [loadmat(feat_path + f_name + '.mat')['feats'] for f_name in sample_feats]
    return y

def get_dict(gd):
    load_data, _ = gd
    _, _, _, worddict = load_data(load_train=False, load_dev=False, load_test=False)
    word_idict = get_word_idict(worddict)
    return worddict, word_idict

