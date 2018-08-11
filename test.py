import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import data
import model as lmModel
import os
import hashlib
from utils import batchify, model_load
from train_utils import evaluate
'''
This code is adapted from https://github.com/salesforce/awd-lstm-lm/
'''

###############################################################################
# Testing code
###############################################################################

def test(args):
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
        print('Done')
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)
        print('Done')

    ntokens = len(corpus.dictionary)
    batch_size = args.batchSize
    val_data = batchify(corpus.valid, batch_size, args)
    test_data = batchify(corpus.test, batch_size, args)

    if not os.path.isfile(args.weightFile):
        print('Pre-trained weight file does not exist. Please check the location: {}'.format(args.weightFile))
        exit()
    model, criterion, _, _ = model_load(args.weightFile)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Run on validation data.
    val_loss = evaluate(args, model, criterion, val_data, ntokens, batch_size)
    print('=' * 89)
    print('| End of Validation | val loss {:5.2f} | val ppl {:8.2f}'.format(
        val_loss, math.exp(val_loss)))
    print('=' * 89)

    # Run on test data.
    test_loss = evaluate(args, model, criterion, test_data, ntokens, batch_size)
    print('=' * 89)
    print('| End of Testing | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Langauge model based on PRU')
    parser.add_argument('--data', type=str, default='data/penn/', help='location of the data corpus')
    parser.add_argument('--batchSize', type=int, default=1, help='Batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--weightFile', type=str, default='./results_PRU/model_nl_3_nh_1200_g_2_k_2.pt', help='Weight file')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False

    # test the trained model (best one on the validation set) on the test set
    test(args)
