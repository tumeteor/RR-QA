# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from .utils import AverageMeter
from .rnn_reader import RnnDocReader

# Modification:
#   - change the logger name
#   - save & load "state_dict"s of optimizer and loss meter
#   - save all random seeds
#   - change the dimension of inputs (for POS and NER features)
#   - remove "reset parameters" and use a gradient hook for gradient masking
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.device = torch.cuda.current_device() if opt['cuda'] else torch.device('cpu')
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        if state_dict:
            self.train_loss.load(state_dict['loss'])

        # Building network.
        self.network = RnnDocReader(opt, embedding=embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
        self.network.to(self.device)

        # Building optimizer.
        self.opt_state_dict = state_dict['optimizer'] if state_dict else None
        self.build_optimizer()

    def build_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.opt['learning_rate'],
                                       momentum=self.opt['momentum'],
                                       weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.opt['optimizer'])
        if self.opt_state_dict:
            self.optimizer.load_state_dict(self.opt_state_dict)

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        inputs = [e.to(self.device) for e in ex[:7]]
        target_s = ex[7].to(self.device) #text
        target_e = ex[8].to(self.device) #span

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.item())

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            # ex (input): context_id, context_feature, context_tag, context_ent, context_mask,
            #           question_id, question_mask, text, span
            inputs = [Variable(e.cuda(async=True)) for e in ex[:7]]
        else:
            inputs = [Variable(e) for e in ex[:7]]

        # Run forward
        with torch.no_grad():
            score_s, score_e = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        pscores = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            # i: each instance in a batch
            # P_start(i) * P_end(i)
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            # get the coordinates
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])
            pscores.append(np.max(scores))

        return (predictions, pscores)

    def predict_cand(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            # ex (input): context_id, context_feature, context_tag, context_ent, context_mask,
            #           question_id, question_mask, text, span
            # IMPORTANT: batchsize must be 1 in this case
            cand_size = len(ex[0])
            inputs = []
            for i in range(0, cand_size):
                x = [Variable(e.cuda(async=True)).unsqueeze(0) for e in (ex[0][i], ex[1][i],
                                                                      ex[2][i], ex[3][i],
                                                                      ex[4][i])]
                x.extend([Variable(e.cuda(async=True))for e in (ex[5], ex[6])])
                inputs.append(x)

        else:
            cand_size = len(ex[0])
            inputs = []
            for i in range(0, cand_size):
                x = [Variable(e).unsqueeze(0) for e in (ex[0][i], ex[1][i],
                                                                      ex[2][i], ex[3][i],
                                                                      ex[4][i])]
                x.extend([Variable(e)for e in (ex[5], ex[6])])
                inputs.append(x)


        # Run forward
        with torch.no_grad():
            pred = {}
            for k in range(0, cand_size):
                input = inputs[k]
                score_s, score_e = self.network(*input)
                # Transfer to CPU/normal tensors for numpy ops
                score_s = score_s.data.cpu()
                score_e = score_e.data.cpu()
                # Get argmax text spans
                text = ex[-2][k] # context (text) list
                spans = ex[-1][k] # context span list

                max_len = self.opt['max_len'] or score_s.size(1)
                for i in range(score_s.size(0)):
                    # i: each instance in a batch
                    # P_start(i) * P_end(i)
                    scores = torch.ger(score_s[i], score_e[i])
                    scores.triu_().tril_(max_len - 1)
                    scores = scores.numpy()
                    # get the coordinates
                    s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
                    try:
                        s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                        pred[tuple(text[i][s_offset:e_offset])] = np.max(scores)
                        print("text {}: {}".format(i, text[i]))
                        print("aa: {}".format(text[i][s_offset:e_offset]))
                        print("spans {}: {}".format(i, spans[i]))
                    except IndexError:
                        pred[tuple("")] = 0



            pred_sorted = sorted(pred, key=pred.get, reverse=True)
            return (list(pred_sorted[0]), pred[pred_sorted[0]])


    def save(self, filename, epoch, scores):
        em, f1, best_eval = scores
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates,
                'loss': self.train_loss.state_dict()
            },
            'config': self.opt,
            'epoch': epoch,
            'em': em,
            'f1': f1,
            'best_eval': best_eval,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')
