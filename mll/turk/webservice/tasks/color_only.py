import uuid
import random

from PIL import Image, ImageDraw

from mll.turk.webservice import task_creator_lib, drawing_lib, cached_grammar


class ColorOnly:
    """
    we need to generate a language, and also images
    perhaps we should generate the whole language, and store it in a file (or in the db),
    and then just load the existing language each time?
    (otherwise, creating a new language each time sounds slow...)
    """
    def __init__(self, seed: int, grammar: str, num_examples: int = 50):
        self.seed = seed
        # self.idx = idx
        self.grammar = grammar
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]
        num_meaning_types = 1
        meanings_per_type = len(self.colors)
        vocab_size = 4
        self.cached_grammar = cached_grammar.CachedGrammar(
            seed=seed, num_meaning_types=num_meaning_types, meanings_per_type=meanings_per_type,
            vocab_size=vocab_size, num_examples=num_examples, grammar=grammar
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
        color = task_creator_lib.perturb_color(color, 40)
        # print('color', color)
        size = random.randint(50, 150) * antialias_multiple
        rotate = random.randint(0, 360)
        left_d = random.randint(-50, 50) * antialias_multiple
        up_d = random.randint(-50, 50) * antialias_multiple
        drawing_lib.draw_polygon_centered(
            draw, left=im_width // 2 * antialias_multiple + left_d, up=im_height // 2 * antialias_multiple + up_d,
            radius=size, sides=4, rotate=rotate, fill=color, outline=color)
        im = im.resize((im_width, im_height), Image.ANTIALIAS)
        filename = uuid.uuid4().hex
        filepath = f'html/img/{filename}.png'
        im.save(filepath)
        return {'filepath': filepath, 'expected': expected_utt, 'meaning': meaning}
