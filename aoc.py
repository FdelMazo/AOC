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

    with open('input-day-one') as f:
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

    with open('input-day-two') as f:
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
