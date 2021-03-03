import torch
import pytest

from mll import darts_cell


@pytest.mark.parametrize(
    "genotype",
    [
        darts_cell.sender_genotype,
        darts_cell.receiver_genotype
    ]
)
def test_darts_cell(genotype):
    N = 3
    ninp = 7
    nhid = 11
    cell = darts_cell.DARTSCell(ninp=ninp, nhid=nhid, genotype=genotype)
    inputs = torch.rand(N, ninp)
    state = torch.zeros(N, nhid)
    new_state = cell(inputs, state)
    print('new_state', new_state)
    assert list(new_state.size()) == [N, nhid]


@pytest.mark.parametrize(
    "genotype",
    [
        darts_cell.sender_genotype,
        darts_cell.receiver_genotype
    ]
)
def test_darts(genotype):
    N = 4
    seq_len = 5

    ninp = 7
    nhid = 11
    num_layers = 3

    darts = darts_cell.DARTS(
        num_layers=num_layers,
        input_size=ninp,
        hidden_size=nhid,
        dropout=0.1,
        genotype=genotype
    )
    inputs = torch.rand(seq_len, N, ninp)
    state = torch.rand(num_layers, N, nhid)
    outputs, state = darts(inputs, state)
    assert list(outputs.size()) == [seq_len, N, nhid]
    assert list(state.size()) == [num_layers, N, nhid]


@pytest.mark.parametrize(
    "genotype",
    [
        darts_cell.sender_genotype,
        darts_cell.receiver_genotype
    ]
)
def test_darts_no_state_input(genotype):
    N = 4
    seq_len = 5

    ninp = 7
    nhid = 11
    num_layers = 3

    darts = darts_cell.DARTS(
        num_layers=num_layers,
        input_size=ninp,
        hidden_size=nhid,
        dropout=0.1,
        genotype=genotype
    )
    inputs = torch.rand(seq_len, N, ninp)
    outputs, state = darts(inputs)
    assert list(outputs.size()) == [seq_len, N, nhid]
    assert list(state.size()) == [num_layers, N, nhid]
