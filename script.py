#!/usr/bin/env python3
import pygal
from csv import DictReader
from pathlib import Path
import sys
from sympy import floor, Float, Rational
from sortedcontainers import SortedList, SortedDict
from pprint import pp
import re
from collections import namedtuple


BOOST_LEN     = 6
QUEUE_MAX_LEN = 6 # Not yet implemented
MAX_ROUND     = 31 # Inclusive

DATA_FILENAME = "Battles 2 Eco"

HEADERS = {
    "Bloon Name"         : True  ,
    "Multiplier"         : False ,
    "First Round"        : True  ,
    "Last Round"         : True  ,
    "Bloon Delay (s)"    : False ,
    "Cooldown (s)"       : True  ,
    "Cost ($)"           : True  ,
    "Eco ($)"            : True  ,
    "Efficiency (Boost)" : False ,
    "Repay (s)"          : False ,
    "Eco Speed ($/s)"    : False ,
    "Drain ($/Boost)"    : False ,
}

Header = type("Header", (), { re.sub(r"\(.*\)", "", h, 1).strip().upper().replace(' ', '_') : h
                         for h in HEADERS.keys() })

SVG_FILENAME_FORMAT = "round_{:02}.svg"
SVG_FOLDER_NAME     = Path(__file__).parent / "svgs"


VALUE_CONV = {
    Header.BLOON_NAME  : (lambda x: x) ,
    Header.FIRST_ROUND : int           ,
    Header.LAST_ROUND  : int           ,
    Header.COST        : int           ,
}

def format_num(x):
    if isinstance(x, Rational):
        x = str(Float(x)).rstrip('0')
        if x[-1] == '.':
            x = x[ : -1 ]
        else:
            x = re.sub(r"(\.\d[1-9]?)\d+", r"\1", x)
    return str(x)

class BloonEco:
    def __init__(self, amt, obj):
        self.obj     = obj
        self.name    = self.obj[Header.BLOON_NAME]
        self.amt     = amt
        self.eco     = self.amt * self.obj[Header.ECO]
        self.include = False

    def __repr__(self):
        members = []
        for name in ("eco", "name", "include"):
            v = format_num(getattr(self, name))
            members.append(f"{name}={v!r}")
        return f'{self.__class__.__name__}({", ".join(members)})'

    def __str__(self):
        return repr(self)

CHART_ARGS = {
    "stroke"          : False ,
    "truncate_legend" : -1    ,
    "include_x_axis"  : True  ,
    "include_y_axis"  : True  ,
}
CHART_OPTS = {
    "title"           : "Round {}"                       ,
    "x_title"         : "$ to be spent in one boost ($)" ,
    "y_title"         : "Eco gained ($)"                 ,
    "value_formatter" : format_num                       ,
}



def get_file_path():
    input_files = tuple(Path().glob(f"{DATA_FILENAME}.*"))

    if len(input_files) > 1:
        output_str = ('\n'
                        + "".join(f"{s}\n" for s in (
                            "Multiple data files of different file extensions were found.",
                            "Input a natural number corresponding to desired data file.",
                        ))
                        + "\n"
                        + "".join(f"{i}\t{s}\n" for i, s in enumerate(input_files, 1))
                        + "\n"
                     )
        while True:
            try:
                i = int(input(output_str))
                if i in range(1, len(input_files) + 1):
                    break
            except ValueError:
                pass
        input_files = input_files[i - 1 : i]

    return input_files[0]



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
        case '_':
            raise NotImplementedError(f"File type of {input_file} not supported.")

    for row in data:
        row[Header.DRAIN]          = Rational(row[Header.COST], row[Header.COOLDOWN]) * 6
        row[Header.ECO_SPEED]      = Rational(row[Header.ECO], row[Header.COOLDOWN])
        row[Header.EFFICIENCY]     = BOOST_LEN * row[Header.ECO_SPEED]

    return SortedList(data, key = lambda r: r[Header.ECO_SPEED])


def naive_equal_lists(a, b):
    return len(a) == len(b) and all(x in b for x in a)

def calculate_points(data, min_round=1, max_round=MAX_ROUND):

    round_points = {}
    BloonData = namedtuple("BloonData", "bloons data")

    for r in range(min_round, max_round + 1):

        bloons = []
        for row in data:
            if r in range(row[Header.FIRST_ROUND], row[Header.LAST_ROUND] + 1):
                bloons.append(row)

        if r > min_round and naive_equal_lists(round_points[r - 1].bloons, bloons):
            round_points[r] = round_points[r - 1]
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

        round_points[r] = BloonData(bloons, drain_race)

    return round_points



def draw_svgs(round_points):
    for r, (bloons, points) in round_points.items():
        chart = pygal.XY(**CHART_ARGS)
        for k, v in CHART_OPTS.items():
            if isinstance(v, str):
                v = v.format(r)
            chart.__dict__[k] = v

        bloon_points = {}
        bloon_major_points = {}
        for bloon in bloons:
            bloon_points[bloon[Header.BLOON_NAME]]       = []
            bloon_major_points[bloon[Header.BLOON_NAME]] = []
        for drain, possible_bloons in points:
            for bloon_eco in possible_bloons:
                bloon_points[bloon_eco.name].append((drain, bloon_eco.eco))
                if bloon_eco.include:
                    bloon_major_points[bloon_eco.name].append((drain, bloon_eco.eco))

        for i in bloon_points.items():
            chart.add(*i)

        chart.render_to_file(SVG_FOLDER_NAME / SVG_FILENAME_FORMAT.format(r))






def main():

    input_file   = get_file_path   ()

    data         = get_basic_data  (input_file)

    round_points = calculate_points(data)

    SVG_FOLDER_NAME.mkdir(parents=True, exist_ok=True)

    svg_paths    = draw_svgs       (round_points)



if __name__ == "__main__":
    main()
