import uuid
import random

from PIL import Image, ImageDraw

from mll.turk.webservice import task_creator_lib, drawing_lib, cached_grammar


class ShapeColor:
    """
    try a couple of attributes, maybe shape and color
    """
    def __init__(self, grammar: str, seed: int, num_examples: int = 50, meanings_per_type: int = 3):
        self.seed = seed
        num_meaning_types = 2
        vocab_size = 4
        self.meanings_per_type = meanings_per_type
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ][:meanings_per_type]
        self.shapes = [
            'circle',
            'triangle',
            'box',
            'pentagon',
            'hexagon'
        ][:meanings_per_type]
        self.cached_grammar = cached_grammar.CachedGrammar(
            num_meaning_types=num_meaning_types, meanings_per_type=meanings_per_type, vocab_size=vocab_size,
            num_examples=num_examples, seed=seed, grammar=grammar
        )
        self.max_cards = self.cached_grammar.num_pairs

    def create_example(self, idx: int):
        expected_utt, meaning = map(self.cached_grammar.get_meaning_utt(idx).__getitem__, ['utt', 'meaning'])
        # print('expected_utt', expected_utt)
        # print('meaning', meaning)

        background = (230, 230, 230)
        antialias_multiple = 3
        im_width = 400
        im_height = 400
        im = Image.new('RGB', (im_width * antialias_multiple, im_height * antialias_multiple), color=background)
        draw = ImageDraw.Draw(im)
        color = self.colors[meaning[0]]
        shape = self.shapes[meaning[1]]
        color = task_creator_lib.perturb_color(color, 40)
        # print('color', color, 'shape', shape)
        size = random.randint(50, 150) * antialias_multiple
        rotate = random.randint(0, 360)
        left_d = random.randint(-50, 50) * antialias_multiple
        up_d = random.randint(-50, 50) * antialias_multiple
        if shape in ['circle']:
            drawing_lib.draw_circle_centered(
                draw, left=im_width // 2 * antialias_multiple + left_d, up=im_height // 2 * antialias_multiple + up_d,
                radius=size, fill=color, outline=color)
        else:
            num_sides = {
                'circle': 1,
                'box': 4,
                'triangle': 3,
                'pentagon': 5,
                'hexagon': 6
            }[shape]
            drawing_lib.draw_polygon_centered(
                draw, left=im_width // 2 * antialias_multiple + left_d, up=im_height // 2 * antialias_multiple + up_d,
                radius=size, sides=num_sides, rotate=rotate, fill=color, outline=color)
        im = im.resize((im_width, im_height), Image.ANTIALIAS)
        filename = uuid.uuid4().hex
        filepath = f'html/img/{filename}.png'
        im.save(filepath)
        return {'filepath': filepath, 'expected': expected_utt, 'meaning': meaning}
