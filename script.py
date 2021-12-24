#!/usr/bin/env python3
from pygal import XY

from csv import DictReader
from pathlib import Path
import sys
from sympy import floor, Float, Rational, Number
from sortedcontainers import SortedList, SortedSet, SortedDict
from pprint import pp
import re
from dataclasses import dataclass, field
import argparse
from logging import basicConfig, getLogger, INFO, DEBUG
from typing import Any, Callable

from yaml import load, Loader

basicConfig()
LOGGER = getLogger(__name__)
CONFIG = None
SOURCE_FILE_PARENT = Path(__file__).parent
DEFAULT_DATA_FILENAMES = []

def select_file(parent, names, exts, prev_input=None):
    if isinstance(names, str):
        names = (names,)
    if isinstance(exts, str):
        exts = (exts,)

    exts = tuple(f"{'.' * int(ext[0] != '.')}{ext}" for ext in exts)

    cur_path = (parent / names[-1]).resolve()

    if prev_input is not None:
        LOGGER.warning(  'No file satisfied the path of "%s".'
                       + '\n\tCurrently searching "%s"...',
                       prev_input, cur_path,
                      )

    input_files = []
    for ext in exts:
        input_files.extend(parent.glob(names[-1] + ext))

    if len(input_files) == 0:
        if names[-1] in {'*', *DEFAULT_DATA_FILENAMES}:
            if parent == SOURCE_FILE_PARENT:
                raise FileNotFoundError(f'No file with name "{cur_path}" was found.')
            return search_files(SOURCE_FILE_PARENT, names[:1], exts, cur_path)
        return search_files(parent, names + [DEFAULT_DATA_FILENAMES], exts, cur_path)

    if len(input_files) > 1:
        output_str = (    '\n'
                        + "".join(f"{s}\n" for s in (
                            f'Multiple data files satisfying "{cur_path}" were found.',
                             "Input a natural number corresponding to desired data file.",
                        ))
                        + '\n'
                        + "".join(f"{i}\t{s}\n" for i, s in enumerate(input_files, 1))
                        + '\n'
                     )
        while True:
            try:
                i = int(input(output_str))
                if i in range(1, len(input_files) + 1):
                    break
            except ValueError:
                pass
        input_files = input_files[i - 1 : i]

    file_to_use = input_files[0]

    if prev_input is not None:
        LOGGER.info('Using file "%s".', str(file_to_use))

    return file_to_use


"""
Header = type("Header", (), { re.sub(r"\(.*\)", "", h, 1).strip().upper().replace(' ', '_') : h
                         for h in HEADERS.keys() })
"""
SVG_FOLDER_NAME     = Path(__file__).parent / "svgs"


"""
VALUE_CONV = {
    Header.BLOON_NAME  : (lambda x: x) ,
    Header.FIRST_ROUND : int           ,
    Header.LAST_ROUND  : int           ,
    Header.COST        : int           ,
}
"""

@dataclass(match_args=False)
class BloonEco:
    eco     : Number | int = field(init=False)
    name    : str          = field(init=False)
    include : bool         = field(init=False, default=False)
    amt     : int          = field(repr=False)
    obj     : dict         = field(repr=False)

    def __post_init__(self):
        self.name = self.obj[Header.BLOON_NAME]
        self.eco  = self.amt * self.obj[Header.ECO]

@dataclass(match_args=False)
class RoundData:
    rounds : list[int, int]
    bloons : list[dict[str, Any]]
    points : list[list[int, SortedList]]


def chart_title(bounds):
    bounds = SortedSet(bounds)
    return f"Round{'s' * (len(bounds) - 1)} {'-'.join(f'{b:02}' for b in bounds)}"
def filename(bounds):
    bounds = SortedSet(bounds)
    return SVG_FOLDER_NAME / f"round{'s' * (len(bounds) - 1)}_{'-'.join(map(str, bounds))}.svg"

def format_num(x, digits=1):
    if isinstance(x, Rational):
        x = str(Float(x)).rstrip('0')
        if x[-1] == '.':
            x = x[ : -1 ]
        else:
            x = re.sub(rf"(\.\d[1-9]{{0, {digits}}})\d+", r"\1", x)
    return str(x)

CHART_OPTS = {
    "title"           : chart_title                     ,
    "x_title"         : "$ to be spent in one boost ($)",
    "y_title"         : "Eco gained ($)"                ,
    "value_formatter" : (lambda x: f"+${format_num(x)}"),
}


def get_file_path():
    return select_file(Path(), args.data_file, ("svg",))



def get_basic_data(input_file):
    data = None

    match input_file.suffix:
        case ".csv":
            with input_file.open('r') as f:
                data = []
                for row in DictReader(f):
                    for k, v in row.items():
                        if HEADERS.get(k):
                            row[k] = VALUE_CONV.get(k, Rational)(v)
                    data.append(row)
                data = tuple(data)
        case _:
            raise NotImplementedError(f"File type of {input_file.suffix!r} not supported.")

    for row in data:
        row[Header.DRAIN]      = Rational(row[Header.COST], row[Header.COOLDOWN]) * 6
        row[Header.ECO_SPEED]  = Rational(row[Header.ECO], row[Header.COOLDOWN])
        row[Header.EFFICIENCY] = BOOST_LEN * row[Header.ECO_SPEED]

    return SortedList(data, key = lambda r: r[Header.ECO_SPEED])


def naive_equal_lists(a, b):
    return len(a) == len(b) and all(x in b for x in a)

def calculate_points(data, min_round=1, max_round=None):

    round_points = []

    for r in range(min_round, max_round + 1):

        bloons = []
        for row in data:
            if r in range(row[Header.FIRST_ROUND], row[Header.LAST_ROUND] + 1):
                bloons.append(row)

        if r > min_round and naive_equal_lists(round_points[-1].bloons, bloons):
            round_points[-1].rounds[1] += 1
            continue

        drain_race = SortedDict()

        def add_race(b):
            for k in range(1, int(floor(BOOST_LEN / b[Header.COOLDOWN])) + 1):
                if k * b[Header.COST] not in drain_race:
                    drain_race[k * b[Header.COST]] = SortedList(key=lambda x: (-x.eco, x.amt / x.obj[Header.COOLDOWN]))
                drain_race[k * b[Header.COST]].add(BloonEco(k, b))

        for bloon in bloons:
            add_race(bloon)

        drain_race = [ [k, v] for k, v in drain_race.items()]

        include = lambda i, b: not (i > 0 and (drain_race[i - 1][1][0].eco >= b.eco
                                               or drain_race[i - 1][1][0].name == b.name))

        for i, (_, v) in enumerate(drain_race):
            if include(i, v[0]):
                drain_race[i][1][0].include = True

        round_points.append(RoundData([r, r], bloons, drain_race))

    return round_points


def new_chart(rounds):
    chart = CHART_TYPE(**CHART_ARGS)
    for k, v in CHART_OPTS.items():
        if k == "title":
            v = v(rounds)
        chart.__dict__[k] = v
    return chart

def draw_svgs(round_points):
    for r in round_points:
        chart = new_chart(r.rounds)

        bloon_points = {}
        bloon_major_points = []
        for bloon in r.bloons:
            bloon_points[bloon[Header.BLOON_NAME]] = []

        for drain, possible_bloons in r.points:
            for bloon_eco in possible_bloons:
                bloon_points[bloon_eco.name].append((drain, bloon_eco.eco))
            if possible_bloons[0].include:
                bloon_major_points.append(drain)

        for i in bloon_points.items():
            chart.add(*i)


        chart.x_labels = [ str(b) for b in bloon_major_points ]
        chart.x_labels_major = chart.x_labels

        chart.xrange = (0, chart.xrange[1])

        chart.render_to_file(SVG_FOLDER_NAME / 3)






def main():
    global CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs='?', default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    LOGGER.setLevel(DEBUG if args.debug else INFO)
    with select_file(SOURCE_FILE_PARENT, "*", "yaml").open('r') as f:
        CONFIG = load(f, Loader)

    pp(CONFIG)

    input_file   = get_file_path   ()

    data         = get_basic_data  (input_file)

    round_points = calculate_points(data)

    SVG_FOLDER_NAME.mkdir(parents=True, exist_ok=True)

    svg_paths    = draw_svgs       (round_points)



if __name__ == "__main__":
    main()
