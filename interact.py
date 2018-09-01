import time
import argparse
import torch
import msgpack
from drqa.model import DocReaderModel
from drqa.utils import str2bool
from prepro import annotate, to_id, init, prepare_test
from train import BatchGen, score

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

parser.add_argument('--test_file', default='HBCP/effect/dev.effect_cand.json',
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

if (args.batch):
    test, test_y = prepare_test(meta['vocab'], meta['vocab_tag'], meta['vocab_ent'], meta['wv_cased'], args)

    batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    predictions = []
    scores = []
    
    for i, batch in enumerate(batches):
        p, score = model.predict(batch)
        predictions.extend(p)
        scores.extend(score)
        #print('> evaluating [{}/{}]'.format(i, len(batches)))
    em, f1 = score(predictions, test_y, evaluation=True)
    print("dev EM: {} F1: {}".format(em, f1))
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


