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
