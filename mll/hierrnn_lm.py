"""
using tolstoy war and peace (or some text, but war and peace is free (from guttenberg project),
and long, and fairly standard english (since modern translation))

we're going to:
- equip an rnn with 5 categorical output heads,
- create a compositional-ish grammar over 5 meanings types
- train the rnn on this grammar, and test it, and see how it does in train and test
- compare with a random grammar (ie each combination of meanings get a unique utterance)

I guess we can just use teacher forcing???

Let's try with teacher forcing for now anyway, since simpler probably

Also, we'll keep utterances all fixed length, so we dont have to think about the
effect of length, terminators, etc
"""
import time

from torch import nn, optim, autograd

from ulfs.stats import Stats
from ulfs import tensor_utils
from ulfs.runner_base_v1 import RunnerBase

from mll import mem_common, nospaces_dataset
from mll import run_mem_recv, run_mem_send


class HierarchicalEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type

        self.embedder = nn.Embedding(vocab_size, embedding_size)
        self.encoder = run_mem_recv.RNNHierarchicalEncoder(
            embedding_size=embedding_size,
            rnn_type=rnn_type
        )
        self.decoder = run_mem_send.RNNHierarchicalDecoder(
            embedding_size=embedding_size,
            rnn_type=rnn_type,
            dropout=dropout
        )
        self.out_linear = nn.Linear(embedding_size, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, utts, decoder_utt_len):
        embs = self.embedder(utts)
        embs = self.drop(embs)
        encoded, enc_stopness = self.encoder(inputs=embs, return_stopness=True)
        utts_logits, dec_stopness = self.decoder(
            state=encoded, utt_len=decoder_utt_len, return_stopness=True)
        utts_logits = self.out_linear(utts_logits)
        return utts_logits, enc_stopness, dec_stopness


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['hier_enc_dec'],
            additional_save_keys=[],
            step_key='episode'
        )

    def setup(self, p):
        self.dataset = nospaces_dataset.Dataset(in_textfile=p.in_textfile)

        self.hier_enc_dec = HierarchicalEncoderDecoder(
            embedding_size=p.embedding_size,
            rnn_type=p.rnn_type,
            vocab_size=self.dataset.vocab_size,
            dropout=p.dropout
        )
        Opt = getattr(optim, p.opt)
        self.opt = Opt(lr=0.001, params=self.hier_enc_dec.parameters())
        self.crit = nn.CrossEntropyLoss(reduction='none')

        self.stats = Stats([
            'episodes_count',
            'train_acc_sum',
            'train_loss_sum'
        ])
        self.start_time = time.time()

    def step(self, p):
        render = self.should_render()
        stats = self.stats

        def forward_batch():
            batch = self.dataset.sample(
                batch_size=p.batch_size,
                encode_len=50,
                decode_len=15
            )
            encode_utts_t = batch['encode_chars_t']
            decode_target_t = batch['decode_chars_t']
            decode_target_lens_t = batch['decode_lens_t']

            utts_out_logits, enc_stopness, dec_stopness = self.hier_enc_dec(
                utts=encode_utts_t, decoder_utt_len=decode_target_t.size(0))
            # print('utts_out_logits.sum().item()', utts_out_logits.sum().item())
            mask = tensor_utils.lengths_to_mask(decode_target_lens_t, utts_out_logits.size(0)).transpose(0, 1).float()

            _, utts_out_pred = utts_out_logits.max(dim=-1)
            correct = (utts_out_pred == decode_target_t).float()
            # print('correct.sum', correct.sum().item())
            correct = mask * correct
            # print('decode_target_lens_t.float()', decode_target_lens_t.float().sum().item())
            acc = correct.sum(dim=0) / decode_target_lens_t.float()
            decode_is_zero = (decode_target_lens_t == 0)
            acc[decode_is_zero.nonzero().view(-1).long()] = 0
            acc = acc.mean().item()
            # print('acc', acc)

            return utts_out_logits, decode_target_t, mask, acc

        self.hier_enc_dec.train()
        utts_out_logits, decode_target_t, mask, acc = forward_batch()
        # print('acc', acc)

        utts_out_logits_flat = utts_out_logits.view(-1, self.dataset.vocab_size)
        decode_target_flat = decode_target_t.view(-1)

        loss_unreduced = self.crit(utts_out_logits_flat, decode_target_flat)
        loss_unreduced = loss_unreduced.view(*list(utts_out_logits.size())[:-1])
        loss_unreduced = mask * loss_unreduced
        loss = loss_unreduced.mean()
        # print('loss', loss.item())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        stats.train_loss_sum += loss.item()
        stats.train_acc_sum += acc
        stats.episodes_count += 1

        if p.model == 'RNNHierarchical':
            self.lower_stopness_sum += self.agent.model.all_lower_stopness[:, 0].data.cpu()

        if render:
            stats = self.stats
            if p.model == 'RNNHierarchical':
                lower_stopness = ((self.lower_stopness_sum / stats.episodes_count) * 100).int()
                print(lower_stopness.view(-1, 4))
                self.lower_stopness_sum.fill_(0)

            log_dict = {
                'sps': stats.episodes_count / (time.time() - self.last_print),
                'train_acc': stats.train_acc_sum / stats.episodes_count,
                'train_loss': stats.train_loss_sum / stats.episodes_count
            }

            self.hier_enc_dec.eval()
            with autograd.no_grad():
                _, _, _, test_acc = forward_batch()

            # meanings = torch.from_numpy(np.random.choice(p.meanings_per_type, (p.batch_size, p.num_meaning_types),
            # replace=True))
            # utts = self.grammar.meanings_to_utterances(meanings)
            # if p.enable_cuda:
            #     meanings, utts = meanings.cuda(), utts.cuda()
            # test_acc = self.agent.eval(utts=utts, meanings=meanings)
            log_dict['test_acc'] = test_acc

            self.print_and_log(
                log_dict,
                formatstr='e={episode} '
                          't={elapsed_time:.0f} '
                          'sps={sps:.0f} '
                          '| train '
                          'loss {train_loss:.3f} '
                          'acc {train_acc:.3f} '
                          '| test {test_acc:.3f} '
            )
            stats.reset()
            self.render_every_seconds = mem_common.adjust_render_time(
                self.render_every_seconds, time.time() - self.start_time)
            if p.terminate_acc is not None and log_dict['train_acc'] >= p.terminate_acc:
                print('reached terminate acc => terminating')
                self.finish = True
                terminate_reason = 'acc'
            if p.terminate_time_minutes is not None and time.time() - self.start_time >= p.terminate_time_minutes * 60:
                print('reached terminate time => terminating')
                self.finish = True
                terminate_reason = 'timeout'
            if self.finish:
                self.res = {
                    'num_parameters': self.agent.num_parameters,
                    'params': p.__dict__,
                    'episode': self.episode,
                    'log_dict': log_dict,
                    'elapsed_time': time.time() - self.start_time,
                    'terminate_reason': terminate_reason
                }


if __name__ == '__main__':
    runner = Runner()
    runner.add_param('--embedding-size', type=int, default=50)
    runner.add_param('--batch-size', type=int, default=128)
    runner.add_param('--model', type=str, default='RNN')
    runner.add_param('--rnn-type', type=str, default='GRU')
    runner.add_param('--dropout', type=float, default=0.1)
    runner.add_param('--num-layers', type=int, default=1)
    runner.add_param('--clip-grad', type=float, default=3.5)
    runner.add_param('--terminate-acc', type=float, help='finish running if reach this acc')
    runner.add_param('--terminate-time-minutes', type=float, help='finish running if reach this elapsed time')
    runner.add_param('--opt', type=str, default='Adam')
    runner.add_param('--in-textfile', type=str, default='~/data/stories/tolstoy_warandpeace.txt')

    runner.parse_args()
    print('runner.args', runner.args)
    runner.setup_base()
    runner.run_base()
