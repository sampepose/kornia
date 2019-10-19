import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToGrayscale:
    def test_rgb_to_grayscale(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert kornia.rgb_to_grayscale(img).shape == (1, height, width)

    def test_rgb_to_grayscale_batch(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert kornia.rgb_to_grayscale(img).shape == \
            (batch_size, 1, height, width)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.rgb_to_grayscale, (img,), raise_exception=True)

    def test_jit(self):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        to_gray = kornia.color.RgbToGrayscale()
        to_gray_jit = torch.jit.script(to_gray)
        assert_allclose(to_gray(img), to_gray_jit(img))
