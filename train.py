import re
import os
import sys
import math
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
from drqa.model import DocReaderModel
from drqa.utils import str2bool
import numpy as np
import six.moves.cPickle as pickle


def main():
    args, log = setup()
    log.info('[Program starts. Loading data...]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    log.info(opt)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(args.model_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        # synchronize random seed
        random.setstate(checkpoint['random_state'])
        torch.random.set_rng_state(checkpoint['torch_state'])
        if args.cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
            log.info('[learning rate reduced by {}]'.format(args.reduce_lr))
        batches = BatchGenCand(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        em, f1 = infer(batches=batches, model=model, log=log, candidateMode=True, dev_y=dev_y)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        if math.fabs(em - checkpoint['em']) > 1e-3 or math.fabs(f1 - checkpoint['f1']) > 1e-3:
            log.info('Inconsistent: recorded EM: {} F1: {}'.format(checkpoint['em'], checkpoint['f1']))
            log.error('Error loading model: current code is inconsistent with code used to train the previous model.')
            exit(1)
        best_val_score = checkpoint['best_eval']
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1
        best_val_score = 0.0


    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGenCand(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.value,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        log.debug('\n')
        # eval
        batches = BatchGenCand(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)

        em, f1 = infer(batches=batches, model=model, log=log, candidateMode=True, dev_y=dev_y)
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch, [em, f1, best_val_score])
            if em > best_val_score:
                best_val_score = em
                copyfile(
                    model_file,
                    os.path.join(args.model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


def setup():
    parser = argparse.ArgumentParser(
        description='Train a Document Reader model.'
    )
    # system
    parser.add_argument('--log_per_updates', type=int, default=3,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--data_file', default='HBCP/effect-listwise/data.msgpack',
                        help='path to preprocessed data file.')
    parser.add_argument('--model_dir', default='models/HBCP/effect-listwise',
                        help='path to store saved models.')
    parser.add_argument('--save_last_only', action='store_true',
                        help='only save the final models.')
    parser.add_argument('--seed', type=int, default=1013,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    # training
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=10)
    parser.add_argument('-rs', '--resume', default='best_model.pt',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                        help='reduce initial (resumed) learning rate by this factor.')
    parser.add_argument('-op', '--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='only applied to SGD.')
    parser.add_argument('-mm', '--momentum', type=float, default=0,
                        help='only applied to SGD.')
    parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                        help='finetune top-x embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--rnn_padding', action='store_true',
                        help='perform rnn padding (much slower but more accurate).')
    # model
    parser.add_argument('--question_merge', default='self_attn')
    parser.add_argument('--doc_layers', type=int, default=3)
    parser.add_argument('--question_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                        help='use pos tags as a feature.')
    parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                        help='use named entity tags as a feature.')
    parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--dropout_rnn', type=float, default=0.4)
    parser.add_argument('--dropout_nn', type=float, default=0.4)
    parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--rnn_type', default='lstm',
                        help='supported types: rnn, gru, lstm')
    parser.add_argument('--listmode', type=str2bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    if args.resume == 'best_model.pt' and not os.path.exists(os.path.join(args.model_dir, args.resume)):
        # means we're starting fresh
        args.resume = ''

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # setup logger
    class ProgressHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            log_entry = self.format(record)
            if record.message.startswith('> '):
                sys.stdout.write('{}\r'.format(log_entry.rstrip()))
                sys.stdout.flush()
            else:
                sys.stdout.write('{}\n'.format(log_entry))

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    return args, log


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def load_data(opt):
    with open('HBCP/effect-listwise/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['ranker'] = True # for ranker component
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    BatchGenCand.pos_size = opt['pos_size']
    BatchGenCand.ner_size = opt['ner_size']
    with open(opt['data_file'], 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train = data['train']
    data['dev'].sort(key=lambda x: len(x[1]))
    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]
    return train, dev, dev_y, embedding, opt


def infer(batches, model, log, candidateMode=True, dev_y=None):
    predictions = []
    # SET RANKING MODE
    if (candidateMode):
        scores = []

        for i, batch in enumerate(batches):
            p, s = model.predict(batch)
            predictions.extend(p)
            scores.extend(s)
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))

        with open("HBCP/effect-all/dev.effect.dict.pkl", "rb") as f:
            cqDict = pickle.load(f)
        cDict = {}
        for i in range(0, len(predictions)):
            qid = cqDict[i]
            if qid in cDict:
                cDict[qid].append(i)
            else:
                cDict[qid] = [i]

        actualPreds, actualAns, actualScores = rankScore(cDict, predictions=predictions, scores=scores)
        em, f1 = score(actualAns, actualPreds)
        log.warning("dev EM: {} F1: {}".format(em, f1))

    else:
        for i, batch in enumerate(batches):
            p, s = model.predict(batch)
            predictions.extend(p)
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))

        em, f1 = score(predictions, dev_y)
        log.warning("dev EM: {} F1: {}".format(em, f1))

    return em, f1

class BatchGen:
    pos_size = None
    ner_size = None

    def __init__(self, data, batch_size, gpu, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # sort by len
        # data = sorted(data, key=lambda x: len(x[1]))
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 10

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            if not self.eval:
                y_s = torch.LongTensor(batch[8])
                y_e = torch.LongTensor(batch[9])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


class BatchGenCand:
    pos_size = None
    ner_size = None

    def __init__(self, data, batch_size, gpu, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # sort by len
        # data = sorted(data, key=lambda x: len(x[1]))
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 10

            cand_size = len(batch[1])
            context_id_list = []
            context_feature_list = []
            context_tag_list = []
            context_ent_list = []
            context_mask_list = []

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                print(i)
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            text = list(batch[6])
            span = list(batch[7])
            if not self.eval:
                y_s = torch.LongTensor(batch[8])
                y_e = torch.LongTensor(batch[9])

            for k in range(0, cand_size):
                context_len = max(len(x) for x in batch[1][k])
                print("context_len :{}".format(context_len))
                context_id = torch.LongTensor(len(batch[1][k]), context_len).fill_(0)
                for i, doc in enumerate(batch[1][k]):
                    context_id[i, :len(doc)] = torch.LongTensor(doc)

                feature_len = len(batch[2][k][0][0])

                context_feature = torch.Tensor(len(batch[1][k]), context_len, feature_len).fill_(0)
                for i, doc in enumerate(batch[2][k]):
                    for j, feature in enumerate(doc):
                        context_feature[i, j, :] = torch.Tensor(feature)

                context_tag = torch.Tensor(len(batch[1][k]), context_len, self.pos_size).fill_(0)
                for i, doc in enumerate(batch[3][k]):
                    for j, tag in enumerate(doc):
                        context_tag[i, j, tag] = 1

                context_ent = torch.Tensor(len(batch[1][k]), context_len, self.ner_size).fill_(0)
                for i, doc in enumerate(batch[4][k]):
                    for j, ent in enumerate(doc):
                        context_ent[i, j, ent] = 1
                context_mask = torch.eq(context_id, 0)
                question_mask = torch.eq(question_id, 0)
                if self.gpu:
                    context_id = context_id.pin_memory()
                    context_feature = context_feature.pin_memory()
                    context_tag = context_tag.pin_memory()
                    context_ent = context_ent.pin_memory()
                    context_mask = context_mask.pin_memory()
                context_id_list.append(context_id)
                context_feature_list.append(context_feature)
                context_tag_list.append(context_tag)
                context_ent_list.append(context_ent)
                context_mask_list.append(context_mask)


            if self.gpu:
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()


            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id_list, context_feature_list, context_tag_list, context_ent_list, context_mask_list,
                       question_id, question_mask, y_s, y_e, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        #print("precision: {}".format(precision))
        #print("recall: {}".format(recall))
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth, evaluation=False):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        #print(p + " " + t)
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    print("number of EM: {}".format(em))
    pre = 100. * em / total
    f1 = 100. * f1 / total
    if evaluation:
      print("precision: {}".format(pre))
      recall = 100. * em / 87
      print("recall: {}".format(recall))
      print("f1: {}".format((2 * pre * recall) / (pre + recall)))
    return pre, f1



def rankScore(cDict, predictions, scores):
    actualPredictions = []
    actualAns = []
    actualScores = []
    for qid in cDict:
        # group by att + docid
        # and get the prediction with max score
        cList = np.array(cDict[qid])
        pScores = [scores[idx] for idx in cList]
        # bestIdx = cList[pScores.index(max(pScores))]
        # fixed! get relative indices from pScores -> get actual from cList
        bestIdx = [cList[idx] for idx in np.array(pScores).argsort()[-1:][::-1]]
        bestP = [predictions[idx] for idx in bestIdx]
        bestS = [scores[idx] for idx in bestIdx]
        ans = qid.split("\t")[-1]
        actualPredictions.append(bestP)
        actualAns.append(ans)
        actualScores.append(bestS)
    return actualPredictions, actualAns, actualScores

if __name__ == '__main__':
    main()

