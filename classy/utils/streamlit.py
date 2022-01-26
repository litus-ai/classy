import random
from typing import Callable, List

import numpy as np


def get_random_color_generator(colors: List[str]) -> Callable[[], str]:

    random.shuffle(colors)

    colors = iter(colors)

    def f():
        try:
            return next(colors)
        except StopIteration:
            return "#%06x" % random.randint(0x000000, 0xFFFFFF)

    return f


def get_md_200_random_color_generator() -> Callable[[], str]:

    # colors taken from https://gist.githubusercontent.com/daniellevass/b0b8cfa773488e138037/raw/d2182c212a4132c0f3bb093fd0010395f927a219/android_material_design_colours.xml
    # md_.*_200
    colors_md_200 = [
        "#EF9A9A",
        "#F48FB1",
        "#CE93D8",
        "#B39DDB",
        "#9FA8DA",
        "#90CAF9",
        "#81D4fA",
        "#80DEEA",
        "#80CBC4",
        "#A5D6A7",
        "#C5E1A5",
        "#E6EE9C",
        "#FFF590",
        "#FFE082",
        "#FFCC80",
        "#FFAB91",
        "#BCAAA4",
        "#EEEEEE",
        "#B0BBC5",
    ]
    return get_random_color_generator(colors_md_200)


def get_md_400_random_color_generator() -> Callable[[], str]:

    # colors taken from https://gist.githubusercontent.com/daniellevass/b0b8cfa773488e138037/raw/d2182c212a4132c0f3bb093fd0010395f927a219/android_material_design_colours.xml
    # md_.*_400
    colors_md_400 = [
        "#EF5350",
        "#EC407A",
        "#AB47BC",
        "#7E57C2",
        "#5C6BC0",
        "#42A5F5",
        "#29B6FC",
        "#26C6DA",
        "#26A69A",
        "#66BB6A",
        "#9CCC65",
        "#D4E157",
        "#FFEE58",
        "#FFCA28",
        "#FFA726",
        "#FF7043",
        "#8D6E63",
        "#BDBDBD",
        "#78909C",
    ]
    return get_random_color_generator(colors_md_400)
