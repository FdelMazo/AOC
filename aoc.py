# /// script
# dependencies = [
#   "numpy",
#   "z3-solver",
# ]
# ///

import re
import numpy as np
import z3
import functools

from itertools import combinations
from collections import Counter
from bisect import bisect_right
from math import sqrt, inf
from functools import reduce

def day_one():
    end = 0 # Times the dial ends at zero
    rot = 0 # Times the dial rotates through zero
    curr = 50

    def invariant(oldcurr, val):
        nonlocal curr, end, rot
        rot += val // 100
        if (curr > 100):
            if (oldcurr != 0): rot += 1
            curr -= 100
        if (curr < 0):
            if (oldcurr != 0): rot += 1
            curr += 100
        if (curr == 0 or curr == 100):
            if (curr == 100): curr = 0
            end +=1

    def left(val):
        nonlocal curr
        oldcurr = curr
        curr = curr - (val % 100)
        invariant(oldcurr, val)

    def right(val):
        nonlocal curr
        oldcurr = curr
        curr = curr + (val % 100)
        invariant(oldcurr, val)

    with open('inputs/1') as f:
        for line in f:
            oldcurr = curr
            if line[0] == 'R':
                right(int(line[1:]))
            else:
                left(int(line[1:]))
            # print(f"{line.strip()}: {oldcurr} -> {curr} -- end: {end}, rot: {rot}")

    print(f"dial ends at zero {end} times")
    print(f"dial rotated through zero {rot} times")
    print(f"dial clicked at zero {end+rot} total times")

def day_two():
    invalid_easy = set()
    invalid_hard = set()

    with open('inputs/2') as f:
        ranges = f.read().split(',')

    def split_n_ways(lst, n):
        l = len(lst)
        if (l % n != 0):
            return []

        r = []
        for i in range(n):
            r.append(lst[ i*l//n : (i+1)*l // n])

        return r

    for r in ranges:
        [a, b] = r.split('-')
        for n in range(int(a), int(b)+1):
            digits = str(n)
            for i in range(1, len(digits)+1):
                parts = split_n_ways(digits, i)
                if (len(parts) != 1 and len(set(parts)) == 1):
                    if (i == 2): invalid_easy.add(n)
                    invalid_hard.add(n)

    print(f"(easy) sum of invalid ids {sum(invalid_easy)}")
    print(f"(hard) sum of invalid ids {sum(invalid_hard)}")

def day_three():
    r = 0
    n = 12 # 2 = easy ; 12 = hard
    with open('inputs/3') as file:
        for line in file:
            digits = [int(d) for d in line.strip()]
            number = ""
            for _ in range(n):
                to_remove = -(n-len(number)-1)
                m = max(digits if to_remove == 0 else digits[:to_remove])
                number += str(m)
                digits = digits[digits.index(m)+1:]

            s = int(number)
            r += s
    print (r)

def day_four():
    matrix = []
    def get_adj(r, c):
        adj = []

        def _get_adj_in_row(row, c, skip_self = False):
            _adj = []
            if (c != 0):
                _adj.append(matrix[row][c-1])

            if (not skip_self):
                _adj.append(matrix[row][c])

            if (c != len(matrix[row]) - 1):
                _adj.append(matrix[row][c+1])

            return _adj

        if r != 0:
            adj.extend(_get_adj_in_row(r - 1, c))

        adj.extend(_get_adj_in_row(r, c, True))

        if r != len(matrix) - 1:
            adj.extend(_get_adj_in_row(r + 1, c))

        return adj

    with open('inputs/4') as file:
        for line in file:
            matrix.append(list(line.strip()))

    removable_rolls_step = 0
    for r, row in enumerate(matrix):
        for c, ch in enumerate(row):
            if ch == '@' and len([x for x in get_adj(r, c) if x == '@']) < 4:
                removable_rolls_step += 1

    print(f"as a first step, {removable_rolls_step} rolls can be removed")

    removable_rolls = 0
    while (removable_rolls_step > 0):
        removable_rolls_step = 0
        for r, row in enumerate(matrix):
            for c, ch in enumerate(row):
                if ch == '@' and len([x for x in get_adj(r, c) if x == '@']) < 4:
                    removable_rolls_step += 1
                    matrix[r][c] = '.'

        removable_rolls += removable_rolls_step

    print(f"in total, {removable_rolls} rolls can be removed")

def day_five():
    fresh = []
    available = []
    ingredients = 0

    is_fresh = True
    with open('inputs/5') as file:
        for line in file:
            if (len(line.strip()) == 0):
                is_fresh = False
                continue

            if (is_fresh):
                [s, f] = line.strip().split('-')
                s, f = int(s), int(f)
                if (f >= s): fresh.append([s, f])
            else:
                available.append(int(line.strip()))

    fresh = sorted(fresh)
    fresh_undup = [fresh[0]]

    for s, f in fresh[1:]:
        top = fresh_undup[-1][1]
        if (s <= top and f > top):
            fresh_undup[-1][1] = f
        elif s > top:
            fresh_undup.append([s, f])

    for ing in available:
        for s, f in fresh:
            if (ing >= s and ing <= f):
                ingredients += 1
                break


    sums = [f - s + 1 for s, f in fresh_undup]

    print(f"{ingredients} available ingredients")
    print(f"{sum(sums)} total fresh ingredients")

def day_six():
    matrix = []
    with open('inputs/6') as file:
        max_number_length = len(max(file.read().split(), key=lambda x: len(x)))

    with open('inputs/6') as file:
        last_line = file.readlines()[-1].strip('\n')
        groups = []
        for ch in last_line:
            if (ch != ' '):
                groups.append(0)
            else:
                groups[-1] = groups[-1] + 1
        groups[-1] = max_number_length

    with open('inputs/6') as file:
        for line in file:
            line = line.strip('\n')
            numbers = []
            i = 0
            for g in groups:
                number = line[i:i+g]
                for _ in range(g - len(number)): number += ' '
                numbers.append(number)
                i+=g+1
            matrix.append([n for n in numbers])

    matrix = list(zip(*matrix))
    res = 0
    simple_matrix = [[op[-1].strip(), *[int(n.strip()) for n in op[:-1]]] for op in matrix]
    for op in simple_matrix:
        if (op[0] == '+'):
            res += reduce(lambda x, y: x + y, op[1:], 0)
        if (op[0] == '*'):
            res += reduce(lambda x, y: x * y, op[1:], 1)

    print(f"without knowing cephalopod math, the answer is {res}")

    res = 0
    def get_cephalod_numbers(m):
        new_m = list(zip(*m))
        return [int(''.join(n).strip()) for n in new_m]

    hard_matrix = [[op[-1].strip(), *get_cephalod_numbers(op[:-1])] for op in matrix]
    for op in hard_matrix:
        if (op[0] == '+'):
            res += reduce(lambda x, y: x + y, op[1:], 0)
        if (op[0] == '*'):
            res += reduce(lambda x, y: x * y, op[1:], 1)

    print(f"knowing cephalopod math, the answer is {res}")

def day_seven():
    with open('inputs/7') as file:
        lines = file.readlines()
        beams = [0 for _ in range(1, len(lines[0]))]
        beams[lines[0].index('S')] = 1
        splits = 0

        for line in lines:
            line = line.strip()
            if '^' not in line:
                # print(''.join([ch if b == 0 else str(b) for ch, b in zip(line, beams)]))
                continue

            splitters_indices = set([i for i, x in enumerate(line) if x == "^"])
            splitters = [1 if i in splitters_indices else 0 for i in range(len(beams))]

            hits = [b if s == 1 else 0 for s, b in zip(splitters, beams)]

            for i, h in list(enumerate(hits)):
                if h:
                    splits += 1
                    beams[i-1] += h
                    beams[i]    = 0
                    beams[i+1] += h

            # print(''.join([ch if b == 0 else str(b) for ch, b in zip(line, beams)]))

    print(f"there were {splits} splits")
    print(f"there are {sum(beams)} total timelines")

def day_eight():
    nodes = []
    with open('inputs/8') as file:
        for line in file:
            nodes.append(tuple(int(n) for n in line.strip().split(',')))

    distances = [[inf for _ in range(len(nodes))] for _ in range(len(nodes))]

    def D(n1, n2):
        x1, y1, z1 = n1
        x2, y2, z2 = n2
        return sqrt((x1 - x2)**2+(y1 - y2)**2+(z1 - z2)**2)

    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if (i != j): distances[i][j] = D(n1, n2)

    def M():
        m_row = min(distances, key=lambda x: min(x))
        r = distances.index(m_row)
        m_col = min(m_row)
        c = m_row.index(m_col)
        return (r, c)

    groups = []
    i = 1
    while True:
        r, c = M()
        distances[r][c] = inf
        distances[c][r] = inf
        n1, n2 = nodes[r], nodes[c]

        hit_groups = [g for g in groups if n1 in g or n2 in g]
        if not len(hit_groups):
            groups.append(set([n1, n2]))
        else:
            merged = set([n1, n2])
            for hg in hit_groups:
                groups.remove(hg)
                merged = merged.union(hg)
            groups.append(merged)

        n = 10
        if (i == n):
            groups = [sorted(g) for g in groups]
            groups = sorted(groups, key=lambda x: -len(x))
            print(f"after connecting {n} junction boxes, this are the top 3 groups lengths {'*'.join([str(len(g)) for g in groups[:3]])} = {reduce(lambda x, y: x*y, [len(g) for g in groups[:3]], 1)}")

        n = 1000
        if (i == n):
            groups = [sorted(g) for g in groups]
            groups = sorted(groups, key=lambda x: -len(x))
            print(f"after connecting {n} junction boxes, this are the top 3 groups lengths {'*'.join([str(len(g)) for g in groups[:3]])} = {reduce(lambda x, y: x*y, [len(g) for g in groups[:3]], 1)}")


        if (len(groups) == 1 and len(groups[0]) == len(nodes)):
            print(f"finally got everything on the same group. last connection: {(n1,n2)} -> x1*x2={n1[0]*n2[0]}")
            break

        i += 1

def day_nine():
    red_tiles = []
    with open('inputs/9') as file:
        for line in file:
            a, b = line.strip().split(',')
            red_tiles.append((int(a), int(b)))

    areas = np.zeros((len(red_tiles), len(red_tiles)))

    def A(a, b):
        x1, y1 = a
        x2, y2 = b
        return (abs(y2-y1)+1) * (abs(x2-x1)+1)

    def M():
        amax = areas.argmax()
        idx = np.unravel_index(amax, areas.shape)
        return areas[idx], idx

    for i, n1 in enumerate(red_tiles):
        for j, n2 in enumerate(red_tiles):
            if (i > j):
                areas[i, j] = A(n1, n2)

    a, (r, c) = M()
    print(f"largest possible rectangle has corners {red_tiles[r]} and {red_tiles[c]}, with area={a}")

    perimeter = set()
    verticals = {}

    for i, n in enumerate(red_tiles):
        adj = red_tiles[i+1] if i < len(red_tiles) - 1 else red_tiles[0]
        x1, y1 = n
        x2, y2 = adj

        if y1 == y2:
            mn, mx = min(x1, x2), max(x1, x2)
            for xx in range(mn, mx+1):
                perimeter.add((xx, y1))
        else:
            mn, mx = min(y1, y2), max(y1, y2)
            # ray-casting crossings (half-open)
            for y in range(mn, mx):
                verticals.setdefault(y, []).append(x1)

            # perimeter (closed)
            for y in range(mn, mx + 1):
                perimeter.add((x1, y))

    for y in verticals: verticals[y].sort()

    def point_in_polygon(x, y):
        xs = verticals.get(y)
        if not xs:
            return False
        return bisect_right(xs, x) % 2 == 1


    def rect_perimeter(n1, n2):
        x1, y1 = n1
        x2, y2 = n2
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))

        # bottom + top
        for x in range(xmin, xmax + 1):
            yield (x, ymin)
            yield (x, ymax)

        # left + right (skip corners)
        for y in range(ymin + 1, ymax):
            yield (xmin, y)
            yield (xmax, y)

    def is_valid(n1, n2):
        for x, y in rect_perimeter(n1, n2):
            if (x, y) in perimeter:
                continue
            if not point_in_polygon(x, y):
                return False
        return True

    is_rect_valid = False
    while not is_rect_valid:
        a, (r, c) = M()
        # print(f"Testing {red_tiles[r]}, {red_tiles[c]} = {a}")
        areas[r, c] = 0
        is_rect_valid = is_valid(red_tiles[r], red_tiles[c])

    print(f"largest possible rectangle, filled with colored tiles, has corners {red_tiles[r]} and {red_tiles[c]}, with area={a}")

def day_ten():
    machines = []
    with open("inputs/10") as file:
        for line in file:
            lights = re.findall(r"\[(.*)\]", line)[0]
            buttons = re.findall(r"\((\d+(?:,\d+)*)\)", line)
            joltage = re.findall(r"\{(.*)\}", line)[0]

            lights = tuple([i for i, l in enumerate(lights) if l == '#'])
            buttons = [[int(i) for i in button.split(',')] for button in buttons]
            joltage = tuple([int(i) for i in joltage.split(',')])
            machines.append((lights, buttons, joltage))

    def check_lights(lights, buttons, _joltage):
        for c in range(1, len(buttons)):
            combs = combinations(buttons, c)
            for comb in combs:
                full_comb = [x for lst in comb for x in lst]
                full_comb = Counter(full_comb)
                odd = [k for k in full_comb if full_comb[k] % 2 == 1]

                if (sorted(odd) == sorted(lights)):
                    return c

    r = sum([check_lights(*m) for m in machines])
    print(f"The fewest total presses required to configure every machine indicator lights is {r}")

    def check_joltage(_lights, buttons, joltage):
        opt = z3.Optimize()
        coefs = [z3.Int(f"c{i}") for i, _ in enumerate(buttons)]
        for c in coefs: opt.add(c >= 0)

        for i, j in enumerate(joltage):
            opt.add(sum([coefs[bi] for bi, b in enumerate(buttons) if i in b]) == j)

        opt.minimize(sum(coefs))
        opt.check()
        model = opt.model()
        return sum([model[c].as_long() for c in model])

    r = sum([check_joltage(*m) for m in machines])
    print(f"The fewest total presses required to configure every machine joltage is {r}")

def day_eleven():
    graph = {}
    with open("inputs/11a") as file:
        for line in file:
            [flowin, flowout] = line.split(':')
            graph[flowin] = [f.strip() for f in flowout.split(' ') if f]

    paths_you_out = [['you']]
    while not all(['out' in p for p in paths_you_out]):
        for p in paths_you_out:
            top = p[-1]
            if (top == 'out'): continue
            flows = graph[top]
            p.append(flows[0])
            if len(flows) > 1:
                for f in flows[1:]:
                    paths_you_out.append([*p, f])

    print(f"There are {len(paths_you_out)} from 'you' to 'out'")

    graph = {}
    with open("inputs/11b") as file:
        for line in file:
            [flowin, flowout] = line.split(':')
            graph[flowin] = [f.strip() for f in flowout.split(' ') if f]
    graph['out'] = []


    @functools.cache
    def calc_paths(s, d):
        if s == d: return 1
        n = 0
        for o in graph[s]:
            n += calc_paths(o, d)
        return n

    # after manual testing, its confirmed dac comes after fft
    paths_svr_fft = calc_paths('svr', 'fft')
    paths_fft_dac = calc_paths('fft', 'dac')
    paths_dac_out = calc_paths('dac', 'out')

    res = paths_svr_fft * paths_fft_dac * paths_dac_out
    print(f"There are {res} paths from 'svr' to 'out' that go through both 'fft' and 'dac'")
