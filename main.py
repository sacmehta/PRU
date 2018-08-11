import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import time
import data
import model as lmModel
import os
import hashlib
from utils import batchify, model_save, model_load
from train_utils import evaluate, train

'''
This code is adapted from https://github.com/salesforce/awd-lstm-lm/
'''

###############################################################################
# Training script
###############################################################################

def trainEvalLM(args):
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    if torch.cuda.is_available():
        args.cuda = True

    ntokens = len(corpus.dictionary)
    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)

    # Build the model and loss function
    model = lmModel.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                             args.dropout, args.tied, g=args.g, k=args.k)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    #compute network parameters
    params = list(model.parameters())
    total_params = np.sum([np.prod(p.size()) for p in params])
    print('\033[1;32;40mTotal parameters (in million):\033[0m\033[1;31;40m {:0.2f} \033[0m\n'.format(total_params / 1e6, 2))

    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    start_epoch = 1
    if args.resume:
        print('Resuming model ...')
        model, criterion, optimizer, start_epoch = model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropout = args.dropout

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        #Create folder for saving model and log files
        args.saveDir += '_' + args.model
        # =====================
        if not os.path.isdir(args.saveDir):
            os.mkdir(args.saveDir)

        save_str = 'nl_' + str(args.nlayers) + '_nh_' + str(args.nhid) + '_g_' + str(args.g) + '_k_' + str(args.k)
        args.save = args.saveDir + '/model_' + save_str + '.pt'

        logFileLoc = args.saveDir + '/logs_' + save_str + '.txt'
        logger = open(logFileLoc, 'w')
        logger.write(str(args))
        logger.write('\n Total parameters (in million): {:0.2f}'.format(
            total_params / 1e6, 2))
        logger.write('\n\n')
        logger.write("\n%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'ppl (tr)', 'ppl (val)'))
        logger.flush()

        best_val_loss = []
        stored_loss = 100000000
        # Loop over epochs.
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(args, model, criterion, optimizer, epoch, train_data, ntokens)

            ### TRAIN WITH ASGD
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss = evaluate(args, model, criterion, val_data, ntokens, eval_batch_size)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
                print('-' * 89)

                logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                    epoch, train_loss, val_loss,
                    math.exp(train_loss), math.exp(val_loss)))
                logger.flush()

                if val_loss < stored_loss:
                    model_save(args.save, model, criterion, optimizer, epoch)
                    print('Saving Averaged (new best validation)')
                    stored_loss = val_loss

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()


            else:
                val_loss = evaluate(args, model, criterion, val_data, ntokens, eval_batch_size)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
                print('-' * 89)

                logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                    epoch, train_loss, val_loss,
                    math.exp(train_loss), math.exp(val_loss)))
                logger.flush()

                if val_loss < stored_loss:
                    model_save(args.save, model, criterion, optimizer, epoch)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if 't0' not in optimizer.param_groups[0] and (
                        len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0.,
                                                 weight_decay=args.wdecay)
                best_val_loss.append(val_loss)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Langauge model based on PRU')
    #=============General hyper-parameters================================================
    parser.add_argument('--data', type=str, default='data/penn/', help='location of the data corpus')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, help='report interval')
    parser.add_argument('--saveDir', type=str, default='./results', help='Directory to save the models')
    parser.add_argument('--resume', type=str, default='', help='path of model to resume')
    #=============Model related hyper-parameters===========================================
    parser.add_argument('--model', type=str, default='PRU', help='type of recurrent networks (LSTM or PRU)')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1400, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3, help='number of RNN layers')
    parser.add_argument('--g', type=int, default=4, help='No. of groups in Grouped Linear Transformation')
    parser.add_argument('--k', type=int, default=2, help='No. of pyramidal levels in Pyramidal Transformation')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
    #======================Training related hyper-parameters==============================
    parser.add_argument('--lr', type=float, default=30, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--epochs', type=int, default=200, help='Max. number of epochs')
    parser.add_argument('--nonmono', type=int, default=5, help='non monotonous interval')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
    args = parser.parse_args()
    args.tied = True

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.model == 'LSTM':
        args.g = 1
        args.k = 1

    print(args)

    # train and evaluate on the validation set
    trainEvalLM(args)
