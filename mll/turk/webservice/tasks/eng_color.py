import uuid
import random

from PIL import Image, ImageDraw

from mll.turk.webservice import task_creator_lib, drawing_lib


class EngColor:
    def __init__(self):
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]
        self.utts = [
            'red',
            'green',
            'blue',
            'yellow',
            'magenta',
            'cyan',
        ]
        self.meanings = [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5]
        ]
        self.max_cards = len(self.utts)

    def create_example(self, idx: int):
        expected_utt = self.utts[idx]
        meaning = self.meanings[idx]
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
