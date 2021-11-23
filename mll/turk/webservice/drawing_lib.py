import math
from typing import Tuple
from PIL import ImageDraw


def draw_box(
        draw: ImageDraw, left: int, bottom: int, size: int,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        outline_width: int = 10):
    draw.rectangle((
        left,                          bottom-size,
        left+size,                     bottom),
        outline)
    draw.rectangle((
            left + outline_width,      bottom - outline_width,
            left+size - outline_width, bottom - size + outline_width),
        fill)


def draw_circle(
        draw: ImageDraw, left: int, bottom: int, size: int,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        outline_width: int = 10):
    draw.ellipse((
        left,                      bottom - size,
        left + size,               bottom),
        outline)
    draw.ellipse((
        left + outline_width,      bottom - size + outline_width,
        left+size - outline_width, bottom - outline_width),
        fill)


def draw_triangle(
        draw: ImageDraw, left: int, bottom: int, size: int,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        outline_width: int = 10):
    size = size * 2 / math.sqrt(3)
    draw.polygon([
        (left, bottom),
        (left + size, bottom),
        (left + size / 2, bottom - size / 2 * math.sqrt(3))
    ], fill=outline)
    draw.polygon([
        (left + outline_width * math.sqrt(3),        bottom - outline_width),  # bottom left
        (left + size - outline_width * math.sqrt(3), bottom - outline_width),  # bottom right
        (left + size / 2,             bottom - (size) / 2 * math.sqrt(3) + outline_width * 2)  # top
    ], fill=fill)


def draw_polygon_centered(
        draw: ImageDraw, left: int, up: int, radius: int, sides: int,
        fill: Tuple[int, int, int], outline: Tuple[int, int, int],
        outline_width: int = 10, rotate: int = 0):
    if sides == 3:
        outline_width = int(outline_width * math.sqrt(3))
    draw.regular_polygon((left, up, radius), n_sides=sides, rotation=rotate, fill=outline)
    draw.regular_polygon((left, up, radius - outline_width), n_sides=sides, rotation=rotate, fill=fill)


def draw_circle_centered(
        draw: ImageDraw, left: int, up: int, radius: int,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        outline_width: int = 10):

    draw.ellipse((
        left - radius,                      up - radius,
        left + radius,               up+radius),
        outline)
    draw.ellipse((
        left-radius + outline_width,      up - radius + outline_width,
        left+radius - outline_width, up+radius - outline_width),
        fill)
