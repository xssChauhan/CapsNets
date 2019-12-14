from layers import ConvLayer
from layers import PrimaryCapsLayer
from layers import DigiCapsLayer

import torch

import pytest

BATCH_SIZE = 1

@pytest.fixture
def conv_layer_input():
    
    return torch.randn(
        BATCH_SIZE, 1, 28, 28
    )


@pytest.fixture
def primarycaps_input():
    return torch.randn([
        1,256,20,20
    ])


@pytest.fixture
def digicaps_input():
    return torch.randn([
        BATCH_SIZE, 1152, 8
    ])


def test_conv_layer(conv_layer_input):
    conv_layer = ConvLayer()

    op = conv_layer(conv_layer_input)

    assert op.shape == torch.Size([1,256,20,20])


def test_primarycaps_layer(primarycaps_input):
    layer = PrimaryCapsLayer()
    op = layer(primarycaps_input)

    assert op.shape == torch.Size([BATCH_SIZE, 1152, 8])


def test_digicaps_layer(digicaps_input):
    
    layer = DigiCapsLayer()
    op = layer(digicaps_input)

    assert op.shape == torch.Size([
        1, 10,16
    ])