import numpy as np
import time
import argparse
import torch
import msgpack
from drqa.model import DocReaderModel
from drqa.utils import str2bool
from prepro import annotate, to_id, init, prepare_test, prepare_test_cand
from train import BatchGen, BatchGenCand, score
import six.moves.cPickle as pickle

import multiprocessing
"""
This script serves as a template to be modified to suit all possible testing environments, including and not limited 
to files (json, xml, csv, ...), web service, databases and so on.
To change this script to batch model, simply modify line 70 from "BatchGen([model_in], batch_size=1, ...)" to 
"BatchGen([model_in_1, model_in_2, ...], batch_size=batch_size, ...)".
"""

parser = argparse.ArgumentParser(
    description='Interact with document reader model.'
)
parser.add_argument('--model-file', default='models/HBCP/effect/best_model.pt',
                    help='path to model file')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument("--batch", type=str2bool, nargs='?',
                    const=True, default=True,
                    help='whether to use batch evaluation.')

parser.add_argument('--test_file', default='HBCP/effect/test.effect.cand.json',
                    help='path to dev file.')
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--wv_cased', type=str2bool, nargs='?',
                    const=True, default=True,
                    help='treat the words as cased or not.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words. '
                         'Otherwise consider question words first.')
parser.add_argument('--sample_size', type=int, default=0,
                    help='size of sample data (for debugging).')
parser.add_argument('--threads', type=int, default=min(multiprocessing.cpu_count(), 16),
                    help='number of threads for preprocessing.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size for multiprocess tokenizing and tagging.')
args = parser.parse_args()


if args.cuda:
    checkpoint = torch.load(args.model_file)
else:
    checkpoint = torch.load(args.model_file, map_location=lambda storage, loc: storage)

state_dict = checkpoint['state_dict']
opt = checkpoint['config']
with open('HBCP/effect/meta.msgpack', 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')
embedding = torch.Tensor(meta['embedding'])
opt['pretrained_words'] = True
opt['vocab_size'] = embedding.size(0)
opt['embedding_dim'] = embedding.size(1)
opt['pos_size'] = len(meta['vocab_tag'])
opt['ner_size'] = len(meta['vocab_ent'])
opt['cuda'] = args.cuda
BatchGen.pos_size = opt['pos_size']
BatchGen.ner_size = opt['ner_size']
model = DocReaderModel(opt, embedding, state_dict)

candidateMode = True

if (args.batch):
    test, test_y = prepare_test(meta['vocab'], meta['vocab_tag'], meta['vocab_ent'], meta['wv_cased'], args)

    batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    predictions = []
    scores = []
    
    for i, batch in enumerate(batches):
        p, s = model.predict(batch)
        predictions.extend(p)
        scores.extend(s)
        print("prediction :{}".format(p))
        print("score: {}".format(s))
        #print('> evaluating [{}/{}]'.format(i, len(batches)))


    if (candidateMode):
        with open("HBCP/effect/test.effect.dict.pkl", "rb") as f:
            cqDict = pickle.load(f)
        assert len(predictions) == len(cqDict)
        cDict = {}
        for i in range(0, len(predictions)):
            qid = cqDict[i]
            if qid in cDict:
                cDict[qid].append(i)
            else:
                cDict[qid] = [i]
        actualPredictions = []
        actualAns = []
        actualScores = []

        for qid in cDict:
            # group by att + docid
            # and get the prediction with max score
            cList = np.array(cDict[qid])
            pScores = [scores[idx] for idx in cList]
            # bestIdx = cList[pScores.index(max(pScores))]
            bestIdx = cList.argsort()[-1:][::-1]
            bestP = [predictions[idx] for idx in bestIdx]
            bestS = [scores[idx] for idx in bestIdx]
            ans = qid.split("\t")[-1]
            actualPredictions.append(bestP)
            actualAns.append(ans)
            actualScores.append(bestS)

        for i in range(0, len(actualAns)):
            print("{}, {}, {}".format(actualPredictions[i], actualScores[i], actualAns[i]))

        em, f1 = score(actualAns, actualPredictions, evaluation=True)
        print("dev EM: {} F1: {}".format(em, f1))

    # attributes = [[63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76],
    #               [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    #               [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    #               [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    #               ]
    # for att in attributes:
    #     pre = [predictions[idx] for idx in att]
    #     tru = [test_y[idx] for idx in att]
    #     em, f1 = score(pre, tru, evaluation=True)
    #     print("test size: {}".format(len(tru)))
    #     print("dev EM: {} F1: {}".format(em, f1))
    #
    # em, f1 = score(predictions, test_y, evaluation=True)
    # print("test size: {}".format(len(test_y)))
    # print("dev EM: {} F1: {}".format(em, f1))
    

else:
    w2id = {w: i for i, w in enumerate(meta['vocab'])}
    tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
    ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}
    init()

    while True:
        id_ = 0
        try:
            while True:
                evidence = input('Evidence: ')
                if evidence.strip():
                    break
            while True:
                question = input('Question: ')
                if question.strip():
                    break
        except EOFError:
            print()
            break
        id_ += 1
        start_time = time.time()
        annotated = annotate(('interact-{}'.format(id_), evidence, question), meta['wv_cased'])
        model_in = to_id(annotated, w2id, tag2id, ent2id)
        model_in = next(iter(BatchGen([model_in], batch_size=1, gpu=args.cuda, evaluation=True)))
        prediction, score = model.predict(model_in)
        end_time = time.time()
        print('Answer: {}, score: {}'.format(prediction, score))
        print('Time: {:.4f}s'.format(end_time - start_time))


