"""
just write from scratch ...

so, we need:
- an image encoder model. we'll use something like a lenet-5 cnn
- sender network, that takes the encoding as input (?)
- receiver network
    - ... and we will dot product the output of the receiver with the various encoded images
"""
import time

import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np

from ulfs import rl_common, tensor_utils,  utils, nn_modules, metrics
from ulfs.runner_base_v1 import RunnerBase
from ulfs.stats import Stats

from data_code.imagenet.dataset import Dataset


image_files = {
    32: '~/data/imagenet/att_synsets/att_images_t.dat',
    64: '~/data/imagenet/att_synsets/att_images_64_t.dat',
    128: '~/data/imagenet/att_synsets/att_images_128_t.dat'
}


class CNN32(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)  # 16
        self.c2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)  # 16
        self.c3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)  # 8
        self.c4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # 8
        self.c5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)  # 4
        self.c6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)  # 2

        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.bn4 = nn.BatchNorm2d(num_features=16)
        self.bn5 = nn.BatchNorm2d(num_features=32)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        input is 3x32x32 images
        """
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = self.bn5(F.relu(self.c5(x)))
        x = self.c6(x)
        x = self.drop(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        return x


class CNN64(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)  # 32
        self.c2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)   # 32
        self.c3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)  # 16
        self.c4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # 16
        self.c5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)   # 8
        self.c6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  # 8
        self.c7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)   # 4
        self.c8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)   # 2

        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.bn4 = nn.BatchNorm2d(num_features=16)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.bn6 = nn.BatchNorm2d(num_features=32)
        self.bn7 = nn.BatchNorm2d(num_features=64)
        # self.bn8 = nn.BatchNorm2d(num_features=64)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        input is 3x32x32 images
        """
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = self.bn5(F.relu(self.c5(x)))
        x = self.bn6(F.relu(self.c6(x)))
        x = self.bn7(F.relu(self.c7(x)))
        x = self.c8(x)
        x = self.drop(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        return x


class CNN128(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)  # 64
        self.c2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)   # 64
        self.c3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)  # 32
        self.c4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # 32
        self.c5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)   # 16
        self.c6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  # 16
        self.c7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)   # 8
        self.c8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # 8
        self.c9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)   # 4
        self.c10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)   # 2

        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.bn4 = nn.BatchNorm2d(num_features=16)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.bn6 = nn.BatchNorm2d(num_features=32)
        self.bn7 = nn.BatchNorm2d(num_features=64)
        self.bn8 = nn.BatchNorm2d(num_features=64)
        self.bn9 = nn.BatchNorm2d(num_features=64)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        input is 3x32x32 images
        """
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = self.bn5(F.relu(self.c5(x)))
        x = self.bn6(F.relu(self.c6(x)))
        x = self.bn7(F.relu(self.c7(x)))
        x = self.bn8(F.relu(self.c8(x)))
        x = self.bn9(F.relu(self.c9(x)))
        x = self.c10(x)
        x = self.drop(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        return x


class LangSenderModel(nn.Module):
    """
    generator model
    we'll use a differentiable teacher-forcing model, with fixed utterance length
    """
    def __init__(
            self, embedding_size, vocab_size, utt_len, rnn_type, num_layers, input_size,
            dropout
    ):
        self.embedding_size = embedding_size
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        super().__init__()
        self.h_in = nn.Linear(input_size, embedding_size)
        if rnn_type == 'SRU':
            from sru import SRU
            RNN = SRU
        else:
            RNN = getattr(nn, f'{rnn_type}')
        rnn_params = {
            'input_size': embedding_size,
            'hidden_size': embedding_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        if rnn_type == 'SRU':
            rnn_params['rescale'] = False
            rnn_params['use_tanh'] = True
        self.rnn = RNN(**rnn_params)
        self.h_out = nn.Linear(embedding_size, vocab_size)
        self.drop = nn.Dropout(dropout)

        # self.params = self.parameters()
        self.opt = optim.Adam(lr=0.001, params=self.parameters())
        # self.crit = nn.CrossEntropyLoss()

    def forward(self, thoughts):
        """
        thoughts ae [N][input_size]
        """
        N, K = thoughts.size()
        embs = self.h_in(thoughts)
        embs = self.drop(embs)
        device = embs.device

        if self.rnn_type in ['SRU']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=device)
            h[0] = embs
        elif self.rnn_type in ['GRU']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=device)
            h[0] = embs
        else:
            raise Exception(f'unrecognized rnn type {self.rnn_type}')

        fake_input = torch.zeros(self.utt_len, N, self.embedding_size, dtype=torch.float32, device=device)
        if self.rnn_type == 'SRU':
            output, state = self.rnn(fake_input, h)
        elif self.rnn_type in ['GRU']:
            output, h = self.rnn(fake_input, h)
        else:
            raise Exception(f'rnn type {self.rnn_type} not recognized')
        utts = self.h_out(output)
        return utts

    def sup_train_batch(self, thoughts, utts):
        utts_logits = self(thoughts)

        # logits = self.model(meanings)
        _, utts_pred = utts_logits.max(dim=-1)
        correct = utts_pred == utts
        acc = correct.float().mean().item()

        crit = nn_modules.GeneralCrossEntropyLoss()
        loss = crit(utts_logits, utts)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), acc


class LangReceiverModel(nn.Module):
    def __init__(
            self, embedding_size, vocab_size, utt_len,
            rnn_type, dropout, num_layers, output_size
    ):
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        # self.num_meaning_types = num_meaning_types
        # self.meanings_per_type = meanings_per_type

        super().__init__()
        self.embedding = nn_modules.EmbeddingAdapter(vocab_size, embedding_size)
        if rnn_type == 'SRU':
            from sru import SRU
            RNN = SRU
        else:
            RNN = getattr(nn, f'{rnn_type}')
        self.rnn = RNN(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers
        )
        self.h_out = nn.Linear(embedding_size, output_size)
        self.opt = optim.Adam(lr=0.001, params=self.parameters())

    def forward(self, utts, do_predict_correct=False):
        embs = self.embedding(utts)
        seq_len, batch_size, embedding_size = embs.size()
        output, state = self.rnn(embs)
        state = state[-1]
        x = self.h_out(state)
        return x

    def sup_train_batch(self, thoughts, utts):
        thoughts_pred = self(utts)
        crit = nn.MSELoss()
        loss = crit(thoughts_pred, thoughts)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        acc = 0  # placeholder
        return loss.item(), acc


class SoftmaxLink(object):
    def __init__(self, p):
        pass

    def sample_utterances(self, utt_probs):
        return utt_probs

    def calc_loss(self, dp):
        batch_size, _ = dp.size()
        crit = nn.CrossEntropyLoss()
        loss = crit(dp, torch.zeros(batch_size, dtype=torch.int64, device=dp.device))
        loss_v = loss.item()
        return loss, loss_v


class RLLink(object):
    def __init__(self, p):
        self.ent_reg = p.ent_reg

    def sample_utterances(self, utt_probs):
        self.s_sender = rl_common.draw_categorical_sample(
            action_probs=utt_probs,
            batch_idxes=None
        )
        utts = self.s_sender.actions.detach()
        return utts

    def calc_loss(self, dp):
        s_recv = rl_common.draw_categorical_sample(
            action_probs=dp,
            batch_idxes=None
        )
        rewards = (s_recv.actions == 0).float()

        # for reporting purposes:
        loss_v = - rewards.mean().item()

        # lets baseline the reward first
        rewards_mean = rewards.mean().item()
        rewards_std = rewards.std().item()
        rewards = rewards - rewards_mean
        if rewards_std > 1e-1:
            rewards = rewards / rewards_std

        rl_loss = self.s_sender.calc_loss(rewards) + s_recv.calc_loss(rewards)
        loss_all = rl_loss
        if self.ent_reg is not None and self.ent_reg > 0:
            ent_loss = 0
            ent_loss -= self.s_sender.entropy * self.ent_reg
            ent_loss -= s_recv.entropy * self.ent_reg
            loss_all += ent_loss

        loss = loss_all
        return loss, loss_v


class Model(nn.Module):
    def __init__(self, sender_cnn, receiver_cnn, lang_sender, lang_receiver):
        super().__init__()
        self.sender_cnn = sender_cnn
        self.receiver_cnn = receiver_cnn
        self.lang_sender = lang_sender
        self.lang_receiver = lang_receiver


class Agent(nn.Module):
    def __init__(self, p, img_embedding_size):
        super().__init__()
        self.p = p
        self.img_embedding_size = img_embedding_size

        self.lang_sender = LangSenderModel(
            embedding_size=p.embedding_size,
            vocab_size=p.vocab_size,
            utt_len=p.utt_len,
            rnn_type=p.rnn_type,
            num_layers=p.num_layers,
            input_size=self.img_embedding_size,
            dropout=p.dropout
        )
        self.lang_receiver = LangReceiverModel(
            embedding_size=p.embedding_size,
            vocab_size=p.vocab_size,
            utt_len=p.utt_len,
            rnn_type=p.rnn_type,
            dropout=p.dropout,
            num_layers=p.num_layers,
            output_size=self.img_embedding_size
        )
        if p.enable_cuda:
            self.lang_sender = self.lang_sender.cuda()
            self.lang_receiver = self.lang_receiver.cuda()
        self.opt_both = optim.Adam(
            lr=0.001,
            params=list(self.lang_sender.parameters()) + list(self.lang_receiver.parameters())
        )

    def state_dict(self):
        return {
            'lang_sender_state': self.lang_sender.state_dict(),
            'lang_receiver_state': self.lang_receiver.state_dict(),
            'opt_both_state': self.opt_both.state_dict()
        }

    def load_state_dict(self, statedict):
        self.lang_sender.load_state_dict(statedict['lang_sender_state'])
        self.lang_receiver.load_state_dict(statedict['lang_receiver_state'])
        self.opt_both.load_state_dict(statedict['opt_both_state'])


class RefTaskGame(object):
    def __init__(self, sender_cnn, receiver_cnn, sender_lang, receiver_lang, link):
        self.sender_cnn = sender_cnn
        self.receiver_cnn = receiver_cnn
        self.sender_lang = sender_lang
        self.receiver_lang = receiver_lang
        self.link = link

    def forward(self, images_t):
        # sender_images_t, receiver_images_t, distractor_images_t = images_t[:, 0], images_t[:, 1], images_t[:, 2:]
        sender_image_enc = self.sender_cnn(images_t[0])
        utt_logits = self.sender_lang(sender_image_enc)
        utt_probs = F.softmax(utt_logits, dim=-1)

        utts = self.link.sample_utterances(utt_probs)

        receiver_lang_enc = self.receiver_lang(utts)  # [N][E]

        receiver_images_t = images_t[1:]
        d = receiver_images_t.size()
        receiver_images_flat_t = tensor_utils.merge_dims(receiver_images_t, 0, 1)  # [M * N][C][H][W]
        receiver_images_enc_flat = self.receiver_cnn(receiver_images_flat_t)   # [M * N][E]
        receiver_images_enc = tensor_utils.split_dim(receiver_images_enc_flat, 0, d[0], d[1])  # [M][N][E]
        receiver_lang_enc_flat_exp = receiver_lang_enc.unsqueeze(0).expand_as(receiver_images_enc)  # [M][N][E]
        receiver_lang_enc_flat_exp = tensor_utils.merge_dims(
            receiver_lang_enc_flat_exp.contiguous(), 0, 1)   # [M * N][E]

        dp_left = receiver_lang_enc_flat_exp.unsqueeze(-2)  # [M * N][1][E]
        dp_right = receiver_images_enc_flat.unsqueeze(-1)   # [M * N][E][1]
        dp = torch.bmm(dp_left, dp_right)     # [M * N][1][1]
        dp = dp.view(d[0], d[1])   # [M][N]
        dp = dp.transpose(0, 1)   # [N][M]
        self.dp = F.softmax(dp, dim=-1)
        _, pred = dp.max(dim=-1)
        acc = (pred == 0).float().mean().item()
        return utts.detach(), acc

    def calc_loss(self):
        loss, loss_v = self.link.calc_loss(self.dp)
        return loss, loss_v


def batched_run_nograd(params, model, inputs, batch_size, input_batch_dim, output_batch_dim):
    """
    assumes N is multiple of batch_size
    """
    p = params
    N = inputs.size(input_batch_dim)
    num_batches = (N + batch_size - 1) // batch_size
    outputs = None
    count = 0
    for b in range(num_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, N)
        count += (b_end - b_start)
        input_batch = inputs.narrow(dim=input_batch_dim, start=b_start, length=b_end - b_start)
        if p.enable_cuda:
            input_batch = input_batch.cuda()
        with torch.no_grad():
            output_batch = model(input_batch).detach().cpu()
        if outputs is None:
            out_size_full = list(output_batch.size())
            out_size_full[output_batch_dim] = N
            outputs = torch.zeros(*out_size_full, dtype=output_batch.dtype, device='cpu')
        out_narrow = outputs.narrow(dim=output_batch_dim, start=b_start, length=b_end - b_start)
        out_narrow[:] = output_batch
    assert count == N
    return outputs


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['teacher'],
            additional_save_keys=[],
            step_key='training_step'
        )

    def setup(self, p):
        if p.seed is not None:
            torch.manual_seed(p.seed)
            np.random.seed(p.seed)
            print('seeding torch and numpy using ', p.seed)

        self.dataset = Dataset(
            image_size=p.image_size, images_file=p.images_file, num_neg=p.num_neg, att_filepath=p.att_filepath)

        CNN = globals()[f'CNN{p.image_size}']
        self.cnn = CNN(dropout=p.dropout)  # share cnn across everything
        self.sender_cnn = self.cnn
        self.receiver_cnn = self.cnn

        if p.enable_cuda:
            self.cnn = self.cnn.cuda()
            # self.teacher = self.teacher.cuda()

        # determine output size from cnn:
        images_t, atts_t = self.dataset.sample_batch(batch_size=2)
        if p.enable_cuda:
            images_t, atts_t = images_t.cuda(), atts_t.cuda()
        with autograd.no_grad():
            enc_images_t = self.sender_cnn(images_t[0])
        _, self.img_embedding_size = enc_images_t.size()
        print('self.img_embedding_size', self.img_embedding_size)

        self.teacher = Agent(p=p, img_embedding_size=self.img_embedding_size)

        # self.stats = Stats([
        #     'episodes_count',
        #     'loss_sum',
        #     'e2e_acc_sum',
        # ])

        Link = globals()[f'{p.link}Link']
        self.link = Link(p=p)

        self.sup_train_N = int(self.dataset.N_train * p.sup_train_frac)
        print('sup_train_N', self.sup_train_N)

    def step(self, p):
        step = self.training_step
        # stats = self.stats
        link = self.link

        sup_idxes = torch.from_numpy(np.random.choice(self.dataset.N_train, self.sup_train_N, replace=False))
        sup_images = self.dataset.train_images[sup_idxes]
        sup_att = self.dataset.train_att_t[sup_idxes]
        if p.enable_cuda:
            sup_images = sup_images.cuda()
            sup_att = sup_att.cuda()

        with autograd.no_grad():
            sup_images_enc = self.cnn(sup_images).detach()

        print('generating teacher utterances...', end='', flush=True)
        self.teacher.lang_sender.eval()
        sup_utts_logits = batched_run_nograd(
            params=p,
            model=self.teacher.lang_sender,
            inputs=sup_images_enc,
            batch_size=p.batch_size,
            input_batch_dim=0,
            output_batch_dim=1
        )
        self.teacher.lang_sender.train()
        print('sup_utts_logits.size()', sup_utts_logits.size())
        _, sup_utts = sup_utts_logits.max(dim=-1)
        print('sup_utts.size()', sup_utts.size())
        print(' ... done')

        student = Agent(p=p, img_embedding_size=self.img_embedding_size)

        # and then train each half supervised on this data
        # sender first...
        sup_sender_epochs = 0
        sup_receiver_epochs = 0
        sup_sender_acc = 0
        sup_receiver_acc = 0
        sup_sender_time = 0
        sup_receiver_time = 0
        if (
                (p.sup_train_acc is not None and p.sup_train_acc > 0)
                or (p.sup_train_steps is not None and p.sup_train_steps > 0)
        ) and (step > 0 or not p.train_e2e):
            for (agent_str, agent) in [('sender', student.lang_sender), ('recv', student.lang_receiver)]:
                print('sup training on' + agent_str)
                _epoch = 0
                _last_print = time.time()
                _start = time.time()
                while True:
                    # N_train = train_meanings.size(0)
                    num_batches = (self.sup_train_N + p.batch_size - 1) // p.batch_size
                    epoch_acc_sum = 0
                    epoch_loss_sum = 0
                    for b in range(num_batches):
                        b_start = b * p.batch_size
                        b_end = b_start + p.batch_size
                        b_end = min(b_end, self.sup_train_N)
                        b_utts = sup_utts[:, b_start:b_end]
                        b_images_enc = sup_images_enc[b_start:b_end]
                        if p.enable_cuda:
                            b_utts = b_utts.cuda()
                            b_images_enc = b_images_enc.cuda()
                        b_loss, b_acc = agent.sup_train_batch(thoughts=b_images_enc, utts=b_utts)
                        epoch_loss_sum += b_loss * (b_end - b_start)
                        epoch_acc_sum += b_acc * (b_end - b_start)
                    epoch_loss = epoch_loss_sum / self.sup_train_N
                    epoch_acc = epoch_acc_sum / self.sup_train_N
                    _epoch += 1
                    _done_training = False
                    if p.sup_train_acc is not None and epoch_acc >= p.sup_train_acc:
                        print('done sup training (reason: acc)')
                        _done_training = True
                    if p.sup_train_steps is not None and _epoch >= p.sup_train_steps:
                        print('done sup training (reason: steps)')
                        _done_training = True
                    if _done_training or time.time() - _last_print >= 30.0:
                        print(f'sup {agent_str} e={_epoch} loss {epoch_loss:.3f} acc={epoch_acc:.3f}')
                        _last_print = time.time()
                    if _done_training:
                        print('done training for agent', agent.__class__.__name__)
                        break
                if agent == student.lang_sender:
                    sup_sender_epochs = _epoch
                    sup_sender_acc = epoch_acc
                    sup_sender_time = time.time() - _start
                    # stats.send_acc_sum += epoch_acc
                elif agent == student.lang_receiver:
                    sup_receiver_epochs = _epoch
                    sup_receiver_acc = epoch_acc
                    sup_receiver_time = time.time() - _start
                    # stats.recv_acc_sum += epoch_acc
                else:
                    raise Exception('invalid agent value')
            print('done supervised training')

        e2e_time = 0
        if p.train_e2e:
            # then train end to end for a bit, as decoder-encoder, looking at reconstruction accuracy
            # we'll do this on the same meanings as we got from the teacher? or different ones? or
            # just rnadomly sampled from everything except heldout?
            # maybe train on everything except holdout?
            last_print = time.time()
            _e2e_start = time.time()
            e2e_stats = Stats([
                'episodes_count',
                'e2e_loss_sum',
                'e2e_acc_sum',
            ])
            epoch = 0
            ref_task_game = RefTaskGame(
                sender_cnn=self.sender_cnn, receiver_cnn=self.receiver_cnn, sender_lang=student.lang_sender,
                receiver_lang=student.lang_receiver, link=link)
            student_params = student.parameters()
            while True:
                images_t, atts_t = self.dataset.sample_batch(batch_size=p.batch_size, training=True)
                if p.enable_cuda:
                    images_t, atts_t = images_t.cuda(), atts_t.cuda()
                # sender_atts_t, receiver_atts_t, distractor_atts_t = atts_t[0], atts_t[1], atts_t[2:]

                _, acc = ref_task_game.forward(images_t)
                loss, loss_v = ref_task_game.calc_loss()

                e2e_stats.e2e_loss_sum += loss_v
                e2e_stats.e2e_acc_sum += acc
                e2e_stats.episodes_count += 1

                student.opt_both.zero_grad()
                loss.backward()
                if p.clip_grad is not None and p.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(student_params, p.clip_grad)
                student.opt_both.step()

                _done_training = False
                if p.e2e_train_acc is not None and acc >= p.e2e_train_acc:
                    print('reached target e2e acc %.3f' % acc, ' => breaking')
                    _done_training = True
                if p.e2e_train_steps is not None and epoch >= p.e2e_train_steps:
                    print('reached target e2e step', epoch, ' => breaking')
                    _done_training = True
                if time.time() - last_print >= self.render_every_seconds or _done_training:
                    holdout_acc_sum = 0
                    holdout_ep_count = 0
                    holdout_rho_sum = 0
                    self.teacher.lang_sender.eval()
                    for batch_image, batch_att in self.dataset.iter_holdout(batch_size=p.batch_size):
                        if p.enable_cuda:
                            batch_image = batch_image.cuda()
                            batch_att = batch_att.cuda()
                        with autograd.no_grad():
                            utts, acc = ref_task_game.forward(batch_image)
                            if utts.dtype == torch.float32:
                                _, utts = utts.max(dim=-1)
                            holdout_acc_sum += acc
                            holdout_ep_count += 1
                            holdout_rho_sum += metrics.topographic_similarity(utts.transpose(0, 1), batch_att[0])
                    self.teacher.lang_sender.train()
                    rho = holdout_rho_sum / holdout_ep_count
                    holdout_acc = holdout_acc_sum / holdout_ep_count
                    acc = e2e_stats.e2e_acc_sum / e2e_stats.episodes_count
                    loss = e2e_stats.e2e_loss_sum / e2e_stats.episodes_count

                    _elapsed_time = time.time() - _e2e_start
                    log_dict = {
                        'record_type': 'e2e',
                        'ilm_epoch': step,
                        'epoch': epoch,
                        'sps': int(epoch / _elapsed_time),
                        'elapsed_time': int(_elapsed_time),
                        'acc': acc,
                        # 'uniq': uniqueness,
                        'holdout_acc': holdout_acc,
                        'rho': rho,
                        'loss': loss
                    }
                    formatstr = (
                        'e2e e={epoch} i={ilm_epoch} '
                        't={elapsed_time:.0f} '
                        'sps={sps:.0f} '
                        'acc={acc:.3f} '
                        'loss={loss:.3f} '
                        # 'uniq={uniq:.3f} '
                        'holdout_acc={holdout_acc:.3f} '
                        'rho={rho:.3f} '
                    )
                    self.print_and_log(log_dict, formatstr=formatstr)

                    e2e_stats.reset()
                    last_print = time.time()
                if _done_training:
                    break
                epoch += 1
            e2e_time = time.time() - _e2e_start
            e2e_acc = acc
            e2e_holdout_acc = holdout_acc
            e2e_rho = rho
            # stats.e2e_acc_sum += acc

            self.teacher = student

        # stats.episodes_count += 1

        if True:
            log_dict = {
                'type': 'ilm',
                'sps': int(step / (time.time() - self.start_time)),
                'elapsed_time': time.time() - self.start_time,
                # 'uniqueness': uniqueness,
                'e2e_time': e2e_time,
                'e2e_acc': e2e_acc,
                'e2e_holdout_acc': e2e_holdout_acc,
                'e2e_rho': e2e_rho,
                'sup_sender_epochs': sup_sender_epochs,
                'sup_receiver_epochs': sup_receiver_epochs,
                'sup_sender_acc': sup_sender_acc,
                'sup_receiver_acc': sup_receiver_acc,
                'sup_sender_time': sup_sender_time,
                'sup_receiver_time': sup_receiver_time,
                # 'send_acc': stats.send_acc_sum / stats.episodes_count,
                # 'recv_acc': stats.recv_acc_sum / stats.episodes_count,
            }

            formatstr = (
                'e={training_step} '
                't={elapsed_time:.0f} '
                'sps={sps:.0f} '
                'sup_snd[e={sup_sender_epochs} acc={sup_sender_acc:.3f} t={sup_sender_time:.0f}] '
                'sup_rcv[e={sup_receiver_epochs} acc={sup_receiver_acc:.3f} t={sup_receiver_time:.0f}] '
                # 'send_acc={send_acc:.3f} '
                # 'recv_acc={recv_acc:.3f} '
                'e2e[acc={e2e_acc:.3f} t={e2e_time:.0f}] '
                # 'uniq={uniqueness:.3f} '
                'holdout_acc={e2e_holdout_acc:.3f} '
                'rho={e2e_rho:.3f} '
            )
            self.print_and_log(log_dict, formatstr=formatstr)
            # stats.reset()
        if p.max_generations is not None and step + 1 >= p.max_generations:
            print('reached max generations', p.max_generations, '=> terminating')
            self.finish = True


if __name__ == '__main__':
    utils.clean_argv()
    runner = Runner()

    # runner.add_param('--images-file', type=str, default='~/data/imagenet/att_synsets/att_images_t.dat')
    runner.add_param('--image-size', type=int, default=32, choices=[32, 64, 128])
    runner.add_param('--att-filepath', type=str, default='~/data/imagenet/attrann.mat')
    runner.add_param('--num-neg', type=int, default=5)
    runner.add_param('--seed', type=int)
    runner.add_param('--batch-size', type=int, default=32)
    runner.add_param('--link', type=str, default='Softmax')
    runner.add_param('--clip-grad', type=float, default=0)
    runner.add_param('--sup-train-acc', type=float)
    runner.add_param('--sup-train-steps', type=int)
    runner.add_param('--e2e-train-acc', type=float)
    runner.add_param('--e2e-train-steps', type=int)
    runner.add_param('--max-generations', type=int)
    runner.add_param('--sup-train-frac', type=float, default=0.4)
    runner.add_param('--no-train-e2e', action='store_true')

    runner.add_param('--embedding-size', type=int, default=50)
    runner.add_param('--vocab-size', type=int, default=4, help='excludes any terminator')
    runner.add_param('--model', type=str, default='RNN')
    runner.add_param('--rnn-type', type=str, default='GRU')
    runner.add_param('--num-layers', type=int, default=1)
    runner.add_param('--dropout', type=float, default=0.5)
    runner.add_param('--utt-len', type=int, default=8)
    runner.add_param('--ent-reg', type=float, default=0)

    runner.parse_args()
    runner.params.images_file = image_files[runner.params.image_size]
    utils.reverse_args(runner.params, 'no_train_e2e', 'train_e2e')
    print('runner.params', runner.params)
    runner.setup_base()
    runner.run_base()
