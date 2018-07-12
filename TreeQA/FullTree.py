import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BinaryTreeLeafModule(nn.Module):

    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)

    def forward(self, input):
        c = self.cx(input)
        o = F.sigmoid(self.ox(input))
        h = o * F.tanh(c)
        return c, h

class BinaryTreeComposer(nn.Module):

    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

    def forward(self, lc, lh , rc, rh):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh))
        update = F.tanh(self.ulh(lh) + self.urh(rh))
        c =  i* update + lf*lc + rf*rc
        h = F.tanh(c)
        return c, h

class BinaryTreeTopDownComposer(nn.Module):

    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.iph = new_gate()
        self.pfph = new_gate()
        self.uph = new_gate()

    def forward(self, pc, ph):
        i = F.sigmoid(self.ilh(ph))
        pf = F.sigmoid(self.lflh(ph))
        update = F.tanh(self.ulh(ph))
        c =  i* update + pf*pc
        h = F.tanh(c)
        return c, h


class BinaryTreeTopDownLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, criterion,bp_lstm):
        super(BinaryTreeTopDownLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.criterion = criterion
        self.bp_lstm = self.bp_lstm

        self.leaf_module = BinaryTreeLeafModule(in_dim, mem_dim)
        self.composer = BinaryTreeTopDownComposer(in_dim, mem_dim)
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, embs, training = False):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        if tree.num_children == 0:
            # leaf case
            # input: embedding + h(p) + c(p)
            pc, ph = self.get_parent_state(tree)
            tree.state = torch.cat((self.leaf_module.forward(embs[tree.idx-1]),
                                    self.leaf_module.forward(ph),
                                    self.leaf_module.forward(pc)),1)
        elif tree.parent is None:
            # root case
            tree.state = self.bp_lstm.root_c, self.bp_lstm.root_h
        else:
            for idx in range(tree.num_children):
                _ = self.forward(tree.children[idx], embs, training)
            pc, ph = self.get_parent_state(tree)
            tree.state = self.composer.forward(pc, ph)

        return tree.state


    def get_parent_state(self, tree):
        if tree.parent is None:
            return None
        else:
            c, h = tree.parent.state
        return c, h

    def get_child_state(self, tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh


class BinaryTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, criterion):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.criterion = criterion

        self.leaf_module = BinaryTreeLeafModule(cuda,in_dim, mem_dim)
        self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)
        self.output_module = None
        self.root_c = None
        self.root_h = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, embs, training = False):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        if tree.num_children == 0:
            # leaf case
            tree.state = self.leaf_module.forward(embs[tree.idx-1])
        else:
            for idx in range(tree.num_children):
                _ = self.forward(tree.children[idx], embs, training)
            lc, lh, rc, rh = self.get_child_state(tree)
            tree.state = self.composer.forward(lc, lh, rc, rh)
            if tree.parent is None:
                # root case
                self.root_c, self.root_h = tree.state

        return tree.state


    def get_child_state(self, tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh



class ChildSumTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        """
        super(ChildSumTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def node_forward(self, inputs, child_c, child_h):
        """"""

        # sum over hidden states of child nodes
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        # TreeLSTM gates computation
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )

        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        """"""
        # iterate over child nodes
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:  # leaf node
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
        else:  # internal node
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)  # tree.idx from 0

        return tree.state


class BinaryTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        """
        super(BinaryTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def node_forward(self, inputs, child_c, child_h):
        """"""

        # sum over hidden states of child nodes
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        # TreeLSTM gates computation
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )

        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        """"""
        # iterate over child nodes
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:  # leaf node
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
        else:  # internal node
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)  # tree.idx from 0

        return tree.state
