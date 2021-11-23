import uuid
import random
import string
import os
from os.path import join

import torch
from PIL import Image, ImageDraw
import numpy as np

from ulfs import tensor_utils
from mll.turk.webservice import task_creator_lib, drawing_lib
from mll import mem_common


def twodstr_to_tensor(utts, vocab=string.ascii_lowercase):
    N = len(utts)
    M = len(utts[0])
    utts_t = torch.zeros((N, M), dtype=torch.int64)
    # print('utts_t 1', utts_t)
    for n, utt in enumerate(utts):
        for j, letter in enumerate(utt):
            utts_t[n, j] = ord(letter) - ord('a')
    # print('utts_t 2', utts_t)
    return utts_t


class EngShapeColor:
    def __init__(self, seed: int, num_examples: int, grammar: str, meanings_per_type: int = 5):
        self.color_names = [
            ('red', (255, 0, 0)),
            ('grn', (0, 255, 0)),
            ('blu', (0, 0, 255)),
            ('yel', (255, 255, 0)),
            ('mag', (255, 0, 255)),
            ('cyn', (0, 255, 255))
        ][:meanings_per_type]
        self.grammar = grammar
        self.shape_sides = [
            ('cir', 1),
            ('tri', 3),
            ('box', 4),
            ('pnt', 5),
            ('hex', 6)
        ][:meanings_per_type]
        self.meanings = []
        self.utts = []
        self.colors = []
        self.num_sides = []
        for i, (color_code, color) in enumerate(self.color_names[:meanings_per_type]):
            for j, (shape_code, num_sides) in enumerate(self.shape_sides[:meanings_per_type]):
                self.meanings.append((i, j))
                self.utts.append(color_code + shape_code)
                self.colors.append(color)
                self.num_sides.append(num_sides)
        # utts_t = torch.tensor(self.utts)

        language_dir = (
            f'data/languages/eng_g{grammar}_s{seed}_n{num_examples}_'
            f'm{meanings_per_type}')
        if not os.path.isdir(language_dir):
            os.makedirs(language_dir)
        if not os.path.isfile(join(language_dir, 'utts.txt')):
            # need to cache this... (and ideally, seed it...)
            r = np.random.RandomState(seed)
            if grammar != 'Compositional':
                self.corruption = mem_common.get_corruption(
                    corruption_name=grammar, vocab_size=26, meanings_per_type=5, num_meaning_types=2,
                    tokens_per_meaning=3, r=r)
                utts_t = twodstr_to_tensor(self.utts, vocab=string.ascii_lowercase)
                utts_t = self.corruption(utts_t)
                self.utts = tensor_utils.tensor_to_2dstr(utts_t, vocab=string.ascii_lowercase).split('\n')
                # print('self.utts', self.utts, len(self.utts), self.utts[0])
            with open(join(language_dir, 'utts.txt'), 'w') as f:
                for utt in self.utts:
                    f.write(utt + '\n')

        self.utts = []
        with open(join(language_dir, 'utts.txt')) as f:
            for line in f:
                self.utts.append(line.strip())

        self.max_cards = len(self.meanings)
        # self.max_cards =

    def create_example(self, idx: int):
        expected_utt = self.utts[idx]
        meaning = self.meanings[idx]
        # print('expected_utt', expected_utt)
        # print('meaning', meaning)
        color = self.colors[idx]
        num_sides = self.num_sides[idx]

        background = (230, 230, 230)
        antialias_multiple = 3
        im_width = 400
        im_height = 400
        im = Image.new('RGB', (im_width * antialias_multiple, im_height * antialias_multiple), color=background)
        draw = ImageDraw.Draw(im)
        color = task_creator_lib.perturb_color(color, 40)
        # print('color', color)
        size = random.randint(50, 150) * antialias_multiple
        rotate = random.randint(0, 360)
        left_d = random.randint(-50, 50) * antialias_multiple
        up_d = random.randint(-50, 50) * antialias_multiple
        if num_sides > 2:
            drawing_lib.draw_polygon_centered(
                draw, left=im_width // 2 * antialias_multiple + left_d, up=im_height // 2 * antialias_multiple + up_d,
                radius=size, sides=num_sides, rotate=rotate, fill=color, outline=color)
        else:
            drawing_lib.draw_circle_centered(
                draw, left=im_width // 2 * antialias_multiple + left_d, up=im_height // 2 * antialias_multiple + up_d,
                radius=size, fill=color, outline=color)
        im = im.resize((im_width, im_height), Image.ANTIALIAS)
        filename = uuid.uuid4().hex
        filepath = f'html/img/{filename}.png'
        im.save(filepath)
        return {'filepath': filepath, 'expected': expected_utt, 'meaning': meaning}
