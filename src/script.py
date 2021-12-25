#!/usr/bin/env python3

from pathlib import Path
from fractions import Fraction
from decimal import Decimal
from math import floor, log10
from sortedcontainers import SortedList, SortedSet, SortedDict
from pprint import pp, pformat
import re
from dataclasses import dataclass, field
import argparse
from logging import basicConfig, getLogger, INFO, DEBUG
from typing import Any, Iterable
from collections import namedtuple
from itertools import chain
from tqdm import tqdm
from io import BytesIO

import yaml
from yaml import Loader

import shelve


from pandas import DataFrame, read_excel

from csv import DictReader

import json

import pickle


from util import DataFileType, ConfigFileType






basicConfig()
LOGGER = getLogger(__name__)
CONFIG = None
SOURCE_FILE_PARENT = Path(__file__).parents[1]
HEADER = None
SVG_FOLDER_NAME = None
MAX_ROUND_DIGITS = None

SCRIPT_DEBUG = None

def my_pformat(paths):
    data = list(map(str, paths))
    compact = pformat(data, compact=True)
    if '\n' in compact:
        compact = f"\n{compact}"
    return compact.replace('\n', "\n\t")

def select_file(parent, names, exts, prev_input=None):
    exts = tuple(f"{'.' * int(ext[0] != '.')}{ext}" for ext in exts)

    cur_paths = tuple((parent / name).resolve() for name in names[-1])

    if prev_input is not None:
        LOGGER.warning('\n\t'.join((   "No file satisfied any of the following names:"
                                    + f" {my_pformat(prev_input)}."                    ,
                                      f"Currently searching {my_pformat(cur_paths)}...",
                                  )))

    input_files = []
    for cur_path in cur_paths:
        if cur_path.suffix in exts and cur_path.is_file():
            input_files.append(cur_path)
        else:
            for ext in exts:
                input_files.extend(parent.glob(cur_path.name + ext))
    input_files = tuple(input_files)

    if len(input_files) == 0:
        default_data_filenames = getattr(CONFIG, "default_data_filenames", frozenset())
        if names[-1] == default_data_filenames or names[-1] == frozenset('*'):
            if parent == SOURCE_FILE_PARENT:
                raise FileNotFoundError(   "A file with any of the following names"
                                        + f" {my_pformat(names[0])} could not be found.")
            return select_file(SOURCE_FILE_PARENT, names[:1], exts, cur_paths)
        return select_file(parent, names + [default_data_filenames], exts, cur_paths)

    if len(input_files) > 1:
        output_str = (    '\n'
                        + "".join(f"{s}\n" for s in (
                             "Multiple data files satisfying some names in"              ,
                             my_pformat(cur_paths)                                       ,
                             "were found."                                               ,
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

    getattr(LOGGER, "debug" if prev_input is None else "info")(f'Using file "{file_to_use!s}".')

    return file_to_use


@dataclass(match_args=False)
class BloonEco:
    eco     : int          = field(init=False)
    name    : str          = field(init=False)
    include : bool         = field(init=False, default=False)
    amt     : int          = field(repr=False)
    obj     : dict         = field(repr=False)

    def __post_init__(self):
        self.name = self.obj[HEADER.BLOON_NAME]
        self.eco  = self.amt * self.obj[HEADER.ECO]

@dataclass(match_args=False)
class RoundData:
    rounds : list[int, int]
    bloons : list[dict[str, Any]]
    points : list[list[int, SortedList]]


def round_str_info(bounds, add_on = lambda x: x):
    as_set = SortedSet(bounds)
    return namedtuple("RoundStrInfo", "s round_str")(
        's' * (len(as_set) - 1),
        '-'.join(add_on(str(b)) for b in as_set)
    )
def chart_title(bounds):
    info = round_str_info(bounds)
    return f"Round{info.s} {info.round_str}"
def filename(bounds):
    info = round_str_info(bounds, lambda x: x.zfill(MAX_ROUND_DIGITS))
    return SVG_FOLDER_NAME / f"round{info.s}_{info.round_str}.svg"

def format_num(x, digits=None):
    if digits is None:
        digits = CONFIG.default_decimal_digits
    if isinstance(x, Fraction):
        x = str(x.numerator / Decimal(x.denominator)).rstrip('0')
        if x[-1] == '.':
            x = x[ : -1 ]
        else:
            digit_re = fr"\d[1-9]{{0, {digits}}}" if digits else ""
            x = re.sub(rf'(\.{digit_re})\d+', r"\1", x)
    return str(x)


def ext_to_enum(input_file, enum):
    suffix = input_file.suffix
    suffix = suffix[int(suffix[0] == '.') : ].lower()

    attr = lambda k, v: suffix in v and getattr(enum, k.upper())

    loop = {
        DataFileType   : lambda i: attr(*i)             ,
        ConfigFileType : lambda i: attr(i.name, i.value),
    }

    for i in (CONFIG.file_exts.items() if enum == DataFileType else enum):
        a = loop[enum](i)
        if a:
            return a

def load_config(config_file):
    global CONFIG, HEADER, SVG_FOLDER_NAME, MAX_ROUND_DIGITS

    match ext_to_enum(config_file, ConfigFileType):
        case ConfigFileType.YAML:
            with config_file.open("rb") as f:
                d = yaml.load(f, Loader)
        case ConfigFileType.SHELVE:
            d = shelve.open(config_file, "r")

    if isinstance(d["default_data_filenames"], str):
        d["default_data_filenames"] = (d["default_data_filenames"],)
    if isinstance(d["default_data_filenames"], Iterable):
        d["default_data_filenames"] = frozenset(map(Path, d["default_data_filenames"]))

    CONFIG = namedtuple("Config", d.keys())(**d)
    HEADER = type("HEADER", (), { re.sub(r"\(.*\)", "", h, 1).strip().upper().replace(' ', '_') : h
                                  for h in CONFIG.headers.keys() })
    SVG_FOLDER_NAME = SOURCE_FILE_PARENT / CONFIG.svg_folder_name
    MAX_ROUND_DIGITS = int(log10(CONFIG.max_round)) + 1



def get_basic_data(input_file):
    data = None

    def interpret_values(row):
        for k, v in row.items():
            if isinstance(v, str) and k in CONFIG.value_conv:
                v = CONFIG.value_conv[k](v)
            row[k] = v

    def csv(file_obj):
        data = []
        for row in DictReader(file_obj):
            interpret_values(row)
            data.append(row)
        file_obj.close()
        return data

    match ext_to_enum(input_file, DataFileType):
        case DataFileType.EXCEL:
            file_obj = BytesIO()
            read_excel(input_file, dtype=object).to_csv(file_obj)
            data = csv(file_obj)
        case DataFileType.CSV:
            data = csv(input_file.open("r"))

        case (DataFileType.JSON | DataFileType.PICKLE) as t:
            modules = {
                DataFileType.JSON   : json  ,
                DataFileType.PICKLE : pickle,
            }
            with input_file.open("rb") as f:
                data = list(modules[t].load(f))
                for row in data:
                    interpret_values(row)

        case _:
            raise NotImplementedError(f"File type of {input_file.suffix!r} not supported.")

    data = tuple(data)

    for row in data:
        row[HEADER.DRAIN]      = Fraction(row[HEADER.COST], row[HEADER.COOLDOWN]) * 6
        row[HEADER.ECO_SPEED]  = Fraction(row[HEADER.ECO], row[HEADER.COOLDOWN])
        row[HEADER.EFFICIENCY] = CONFIG.boost_len * row[HEADER.ECO_SPEED]

    with (SOURCE_FILE_PARENT / "test.pickle").open("wb") as f:
        pickle.dump(data, f)

    return SortedList(data, key = lambda r: r[HEADER.ECO_SPEED])


def naive_equal_lists(a, b):
    return len(a) == len(b) and all(x in b for x in a)

def calculate_points(data, min_round=1, max_round=None):

    if max_round is None:
        max_round = CONFIG.max_round

    round_points = []

    for r in range(min_round, max_round + 1):

        bloons = []
        for row in data:
            if r in range(row[HEADER.FIRST_ROUND], row[HEADER.LAST_ROUND] + 1):
                bloons.append(row)

        if r > min_round and naive_equal_lists(round_points[-1].bloons, bloons):
            round_points[-1].rounds[1] += 1
            continue

        drain_race = SortedDict()

        def add_race(b):
            for k in range(1, int(floor(CONFIG.boost_len / b[HEADER.COOLDOWN])) + 1):
                if k * b[HEADER.COST] not in drain_race:
                    drain_race[k * b[HEADER.COST]] = SortedList(
                            key=lambda x: (-x.eco, x.amt / x.obj[HEADER.COOLDOWN])
                    )
                drain_race[k * b[HEADER.COST]].add(BloonEco(k, b))

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
    chart = CONFIG.chart_type(**CONFIG.chart_args)
    for k, v in CONFIG.chart_opts.items():
        setattr(chart, k, (lambda x: v.format(format_num(x))) if k == "value_formatter" else v)

    chart.title = chart_title(rounds)
    return chart

def draw_svgs(round_points):
    svg_paths = []

    for r in tqdm(round_points, ncols=CONFIG.output_max_cols, unit="file", colour="green"):
        chart = new_chart(r.rounds)

        bloon_points = {}
        bloon_major_points = []
        for bloon in r.bloons:
            bloon_points[bloon[HEADER.BLOON_NAME]] = []

        for drain, possible_bloons in r.points:
            for bloon_eco in possible_bloons:
                bloon_points[bloon_eco.name].append((drain, bloon_eco.eco))
            if possible_bloons[0].include:
                bloon_major_points.append(drain)

        for i in bloon_points.items():
            chart.add(*i)

        """
        chart.x_labels = [ str(b) for b in bloon_major_points ]
        chart.x_labels_major = chart.x_labels

        chart.xrange = (0, chart.xrange[1])
        """
        svg_paths.append(filename(r.rounds))
        if SCRIPT_DEBUG:
            tqdm.write(' '.join(("Overwritten" if svg_paths[-1].is_file() else "Written",
                                 str(svg_paths[-1])                                     ,
                               )))

        chart.render_to_file(svg_paths[-1])

    return svg_paths





def main():
    global SCRIPT_DEBUG

    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs='*', type=Path)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-c", "--config", default=SOURCE_FILE_PARENT / "config.yaml", type=Path)

    args = parser.parse_args()

    SCRIPT_DEBUG = args.debug

    LOGGER.setLevel(DEBUG if SCRIPT_DEBUG else INFO)

    load_config(args.config)

    if len(args.data_file) == 0:
        args.data_file = CONFIG.default_data_filenames
    args.data_file = [frozenset(args.data_file)]

    input_file   = select_file     (Path(), args.data_file, chain(*CONFIG.file_exts.values()))

    data         = get_basic_data  (input_file)

    round_points = calculate_points(data)

    SVG_FOLDER_NAME.mkdir(parents=True, exist_ok=True)

    svg_paths    = draw_svgs       (round_points)





if __name__ == "__main__":
    main()
