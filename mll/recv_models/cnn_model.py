from torch import nn
from ulfs import nn_modules


class CNNModel(nn.Module):
    supports_dropout = True

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, dropout):
        """
        ignores rnn_type
        """
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type

        super().__init__()
        self.embedding = nn_modules.EmbeddingAdapter(vocab_size + 1, embedding_size)
        layers = []
        length = utt_len
        num_conv_blocks = 0
        while length > 4:
            layers.append(nn.Conv1d(embedding_size, embedding_size, 3, padding=1))
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.ReLU())
            length //= 2
            num_conv_blocks += 1
        print('num conv blocks', num_conv_blocks)
        self.cnn = nn.Sequential(*layers)
        print('self.cnn', self.cnn)
        self.h_out = nn.Linear(embedding_size * length, num_meaning_types * meanings_per_type)
        self.drop = nn.Dropout(dropout)
        print(self.h_out)

    def forward(self, utts):
        batch_size = utts.size(1)
        embs = self.embedding(utts)
        # we want this to be [N][E][M]  (embedding corrresponds to channels I guess?)
        embs = embs.transpose(0, 1).transpose(1, 2).contiguous()
        embs = self.drop(embs)
        state = self.cnn(embs)
        state = self.drop(state)
        state = state.view(batch_size, -1)
        x = self.h_out(state)
        view_list = [self.num_meaning_types, self.meanings_per_type]
        x = x.view(batch_size, *view_list)
        return x
