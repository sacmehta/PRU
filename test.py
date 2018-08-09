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
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    ntokens = len(corpus.dictionary)
    batch_size = 1
    val_data = batchify(corpus.valid, batch_size, args)
    test_data = batchify(corpus.test, batch_size, args)

    global model, criterion, optimizer
    model = lmModel.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                             args.dropout, args.tied, g=args.g, k=args.k)
    criterion = nn.CrossEntropyLoss()

    # Load the model weights.
    if not os.path.isfile(args.weightFile):
        print('Pre-trained weight file does not exist. Please check the location: {}'.format(args.weightFile))
        exit()
    model, criterion, optimizer, _ = model_load(args.weightFile)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    if 't0' in optimizer.param_groups[0]:
        tmp = {}
        for prm in model.parameters():
            tmp[prm] = prm.data.clone()
            prm.data = optimizer.state[prm]['ax'].clone()

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
    parser.add_argument('--model', type=str, default='PRU', help='type of recurrent net (LSTM, PRU)')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1200, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3, help='number of RNN layers')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--g', type=int, default=2, help='No. of groups in Grouped Linear Transformation')
    parser.add_argument('--k', type=int, default=2, help='No. of pyramidal levels in Pyramidal Transformation')
    parser.add_argument('--weightFile', type=str, default='./results_PRU/model_nl_3_nh_1200_g_2_k_2.pt', help='Weight file')

    args = parser.parse_args()
    args.tied = True

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        args.cuda = True
        torch.cuda.manual_seed(args.seed)
    else:
        args.cuda = False

    # test the trained model (best one on the validation set) on the test set
    test(args)
