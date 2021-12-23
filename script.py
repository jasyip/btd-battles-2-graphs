#!/usr/bin/env python3
import pygal
from csv import DictReader
from pathlib import Path
import sys
from sympy import floor, Float, Min, Rational
from sympy.abc import x
from sortedcontainers import SortedList, SortedDict
from pprint import pp
import re
from inspect import getmembers


DATA_FILENAME = "Battles 2 Eco"
BOOST_LEN = 6
QUEUE_MAX_LEN = 6
MAX_ROUND = 31 # Inclusive

HEADERS = (
            "Bloon Name",
            "Multiplier",
            "First Round",
            "Last Round",
            "Bloon Delay (s)",
            "Cooldown (s)",
            "Cost ($)",
            "Eco ($)",
            "Efficiency (Boost)",
            "Repay (s)",
            "Eco Speed ($/s)",
            "Drain ($/Boost)",
          )

Header = type("Header", (), { re.sub(r"\(.*\)", "", h, 1).strip().upper().replace(' ', '_') : h
                         for h in HEADERS })

VALUE_CONV = {
    Header.BLOON_NAME : (lambda x: x),
    Header.FIRST_ROUND : int,
    Header.LAST_ROUND : int,
    Header.COST : int,
}

class BloonEco:
    def __init__(self, amt, obj):
        self.obj = obj
        self.name = self.obj[Header.BLOON_NAME]
        self.amt = amt
        self.eco = self.amt * self.obj[Header.ECO]
        self.include = False

    def __repr__(self):
        members = []
        for name in ("eco", "name", "include"):
            v = getattr(self, name)
            if isinstance(v, Rational):
                v = Float(v, dps=2)
            members.append(f"{name}={v!r}")
        return f'{self.__class__.__name__}({", ".join(members)})'

    def __str__(self):
        return repr(self)

def main():

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

    input_file = input_files[0]
    del input_files

    data = None

    match input_file.suffix:
        case ".csv":
            with input_file.open('r') as f:
                data = []
                for row in DictReader(f):
                    for k, v in row.items():
                        if k not in {Header.BLOON_NAME}:
                            row[k] = VALUE_CONV.get(k, Rational)(v)
                    data.append(row)
                data = tuple(data)
        case '_':
            raise NotImplementedError(f"File type of {input_file} not supported.")

    for row in data:

        row[Header.DRAIN] = Rational(row[Header.COST], row[Header.COOLDOWN]) * 6
        row[Header.ECO_SPEED] = Rational(row[Header.ECO], row[Header.COOLDOWN])
        row[Header.EFFICIENCY] = BOOST_LEN * row[Header.ECO_SPEED]

        money_spent = Min(row[Header.DRAIN] * BOOST_LEN, x)
        rounds = floor(money_spent / row[Header.COST])
        row["drain_func"] = rounds * row[Header.COST]
        row["drain_leftover_func"] = money_spent - row["drain_func"]
        row["eco_func"] = rounds * row[Header.ECO]

    data = SortedList(data, key = lambda r: r[Header.ECO_SPEED])

    # round_drain_rankings = [SortedDict()] * (MAX_ROUND + 1)
    round_eco_rankings = [None] * (MAX_ROUND + 1)


    for r in range(10, 11):

        bloons = []
        for row in data:
            if r in range(row[Header.FIRST_ROUND], row[Header.LAST_ROUND] + 1):
                bloons.append(row)

        """
        if i > 1 and set(round_eco_rankings[i - 1].values()) == set(bloons):
            round_eco_rankings[i] = round_eco_rankings[i - 1]
            continue
        """

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

        pp(drain_race)



    """
    #for i in range(1, MAX_ROUND + 1):
    for i in range(10, 11):
        bloons = []
        starting_best = None
        comp = lambda r: (r[Header.COST], -r[Header.ECO])
        for row in data:
            if i in range(row[Header.FIRST_ROUND], row[Header.LAST_ROUND] + 1):
                bloons.append(row)
                if not starting_best or comp(starting_best) > comp(row):
                    starting_best = row

        def swap(a, b):
            i, j = a if isinstance(a, int) else bloons.index(a), b if isinstance(b, int) else bloons.index(b)
            bloons[i], bloons[j] = bloons[j], bloons[i]

        swap(starting_best, 0)

        pp(bloons)

        for i in range(len(bloons)):
            next_best = None
            for j, bloon in enumerate(bloons[i + 1 : ], i + 1):
                if (bloon[Header.COST] > bloons[i][Header.COST]
                    and bloon[Header.ECO_SPEED] > bloons[i][Header.ECO_SPEED]
                   ):
                    eco_race = SortedDict(key=lambda x: -x)
                    drain_race = SortedDict(key=lambda x: -x)


                    add_race(bloons[i])
                    add_race(bloon)

                    for k, v in eco_race.items():





    """



if __name__ == "__main__":
    main()
