import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


sender_genotype = argparse.Namespace()
sender_genotype.recurrent = [
    ('identity', 0),
]
sender_genotype.concat = [1]


receiver_genotype = argparse.Namespace()
receiver_genotype.recurrent = [
    ('tanh', 0),
    ('tanh', 0),
    ('identity', 0),
    ('relu', 2),
]
receiver_genotype.concat = [1, 2, 4, 3]


class DGSender(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int,
        num_layers: int = 1, dropout: float = 0
    ):
        super().__init__()
        self.darts = DARTS(
            genotype=sender_genotype, input_size=input_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout)

    def forward(self, inputs, state=None):
        return self.darts(inputs, state)


class DGReceiver(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int,
        num_layers: int = 1, dropout: float = 0
    ):
        super().__init__()
        self.darts = DARTS(
            genotype=receiver_genotype, input_size=input_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout)

    def forward(self, inputs, state=None):
        return self.darts(inputs, state)


class DARTS(nn.Module):
    def __init__(
            self, input_size: int, hidden_size: int, genotype,
            init_range: float = 0.04,
            num_layers: int = 1, dropout: float = 0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        for n in range(num_layers):
            layer = DARTSCell(
                ninp=input_size if n == 0 else hidden_size,
                nhid=hidden_size,
                genotype=genotype,
                init_range=init_range
            )
            self.layers.append(layer)

    def forward(self, x, state=None):
        """
        inputs:
        - x: [seq_len, batch_size, input_size]
        - state: [num_layers, batch_size, hidden_size]

        outputs:
        output, state
        where:
        - output: [seq_len, batch_size, hidden_size]
        - state: [num_layers, batch_size, hidden_size]

        applies dropout after each output, except for final layer
        """
        seq_len, batch_size, _ = x.size()
        inputs = [input_.squeeze(0) for input_ in x.split(1)]
        if state is None:
            state = [torch.zeros(
                batch_size, self.hidden_size, device=x.device, dtype=torch.float32)
                for layer_idx in range(self.num_layers)]
        else:
            state = [state_.squeeze(0) for state_ in state.split(1)]
        for layer_idx, layer in enumerate(self.layers):
            layer_out = []
            for t in range(seq_len):
                new_state = layer(inputs[t], state[layer_idx])
                state[layer_idx] = new_state
                layer_out.append(new_state)
            if layer_idx + 1 < len(self.layers):
                layer_out = [self.drop(out) for out in layer_out]
            inputs = layer_out
        outputs = torch.stack(layer_out)
        state = torch.stack(state)
        return outputs, state


# code below this line forked from
# https://github.com/gautierdag/cultural-evolution-engine/blob/54ea8d374ff4345c05f03eccfb2e93161e16a050/model/DARTSCell.py
# 2021 feb 10, which code was made available under the MIT License
# https://github.com/gautierdag/cultural-evolution-engine/blob/54ea8d374ff4345c05f03eccfb2e93161e16a050/LICENSE
#
def identity(x):
    return x


class DARTSCell(nn.Module):
    def __init__(self, ninp, nhid, genotype, init_range=0.04):
        super(DARTSCell, self).__init__()
        self.nhid = nhid
        self.genotype = genotype

        # In genotype is None when doing arch search
        steps = len(self.genotype.recurrent)
        self._W0 = nn.Parameter(
            torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-init_range, init_range)
        )  # [ninp + nhid][2 * nhid]
        self._Ws = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(nhid, 2 * nhid).uniform_(-init_range, init_range)
                )
                for i in range(steps)
            ]
        )

        # initialization
        nn.init.xavier_uniform_(self._W0)
        for p in self._Ws:
            nn.init.xavier_uniform_(p)

    def _get_activation(self, name):
        if name == "tanh":
            f = torch.tanh
        elif name == "relu":
            f = F.relu
        elif name == "sigmoid":
            f = torch.sigmoid
        elif name == "identity":
            f = identity
        else:
            raise NotImplementedError
        return f

    def _compute_init_state(self, x, h_prev):
        """
        x: [batch_size][ninp]
        h_prev: [batch_size][nhid]
        """
        xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def forward(self, x, h_prev):
        """
        Inputs:
        - x: [batch_size][ninp]
        - h_prev: [batch_size][nhid]

        Returns:
        - outputs: [batch_size][nhid]
        """
        s0 = self._compute_init_state(x, h_prev)

        states = [s0]
        for i, (name, pred) in enumerate(self.genotype.recurrent):
            s_prev = states[pred]
            ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(name)
            h = fn(h)
            s = s_prev + c * (h - s_prev)
            states += [s]
        output = torch.mean(
            torch.stack([states[i] for i in self.genotype.concat], -1), -1
        )
        return output


class DGReceiverCell(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int,
    ):
        super().__init__()
        self.cell = DARTSCell(
            genotype=receiver_genotype, ninp=input_size,
            nhid=hidden_size)

    def forward(self, inputs, state):
        return self.cell(inputs, state)


class DGSenderCell(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int,
    ):
        super().__init__()
        self.cell = DARTSCell(
            genotype=sender_genotype, ninp=input_size,
            nhid=hidden_size)

    def forward(self, inputs, state):
        return self.cell(inputs, state)
