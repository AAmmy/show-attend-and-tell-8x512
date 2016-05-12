import cPickle as pkl
import gzip
import os
import sys
import time

import numpy

def prepare_data(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    # x: a list of sentences
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if w in worddict and worddict[w] < n_words else 1 for w in str(cc[0]).split()])
        feat_list.append(features[cc[1]])

    lengths = [len(s) for s in seqs]

    if maxlen != None and numpy.max(lengths) >= maxlen:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    #y = numpy.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
    #for idx, ff in enumerate(feat_list):
    #y[idx,:] = numpy.array(ff)
    #y = numpy.array(feat_list).reshape([len(feat_list), 8, 512]).astype('float32')
    y = numpy.zeros((len(feat_list), len(feat_list[0]))).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = numpy.array(ff)
    y = y.reshape([y.shape[0], 8, 512])
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    # import ipdb
    # ipdb.set_trace()
    return x, x_mask, y

def load_data(load_train=True, load_dev=True, load_test=True, path='../coco/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    if load_train:
        train = (cap[name[0]], feat[name[0]])
    else:
        train = None
    if load_test:
        test = (cap[name[2]], feat[name[2]])
    else:
        test = None
    if load_dev:
        valid = (cap[name[1]], feat[name[1]])
    else:
        valid = None

    return train, valid, test, worddict


import json
from scipy.io import loadmat
cap_file = json.load(open('../../Data/coco/dataset.json', 'rb'))['images']
feat_file = loadmat("../../Data/coco/vgg_feats.mat")['feats'].transpose()
cap_ = [[' '.join(c['tokens']) for c in cf['sentences']] for cf in cap_file]
name = ['train', 'dev', 'test']
num = [110000, 5000, 5000]
cap = { name[0]:cap_[0:num[0]],
        name[1]:cap_[num[0]:num[0] + num[1]],
        name[2]:cap_[num[0] + num[1]:num[0] + num[1] + num[2]]}
for n in name:
    cap[n] = [[[c, cs[0]] for c in cs[1]] for cs in enumerate(cap[n])] # add lookup
    cap[n] = [c for cs in cap[n] for c in cs] # flatten
    
feat = {}
feat = {name[0]:feat_file[0:num[0]],
        name[1]:feat_file[num[0]:num[0] + num[1]],
        name[2]:feat_file[num[0] + num[1]:num[0] + num[1] + num[2]]}


from collections import OrderedDict
dict_count = {}
for cs in cap_:
    for c in cs:
        for w in c.split():
            if w in dict_count:
                dict_count[w] += 1
            else:
                dict_count[w] = 1
dict_ordered_by_count = OrderedDict(sorted(dict_count.items(), key = lambda x:x[0]))
dict_ordered_by_count = OrderedDict(sorted(dict_ordered_by_count.items(), key = lambda x:-x[1]))
worddict = {k:i+2 for i,k in enumerate(dict_ordered_by_count.keys())}


