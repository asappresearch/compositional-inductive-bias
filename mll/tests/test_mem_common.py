from mll import mem_common,  corruptions_lib


def test_shuffle_words():
    num_meaning_types = 3
    tokens_per_meaning = 4
    meanings_per_type = 5
    vocab_size = 7
    corruptions = None

    grammar = mem_common.CompositionalGrammar(
        num_meaning_types=num_meaning_types,
        tokens_per_meaning=tokens_per_meaning,
        meanings_per_type=meanings_per_type,
        vocab_size=vocab_size,
        corruptions=corruptions
    )
    for n in range(30):
        print(grammar.utterances_by_meaning[n])
    corruption = corruptions_lib.ShuffleWordsCorruption(
        num_meaning_types=num_meaning_types,
        tokens_per_meaning=tokens_per_meaning,
        vocab_size=vocab_size,
    )
    corruption(grammar.utterances_by_meaning)
    for n in range(30):
        print(grammar.utterances_by_meaning[n])
