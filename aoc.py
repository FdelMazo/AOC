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

    with open('input-day-one') as input:
        for line in input:
            oldcurr = curr
            if line[0] == 'R':
                right(int(line[1:]))
            else:
                left(int(line[1:]))
            # print(f"{line.strip()}: {oldcurr} -> {curr} -- end: {end}, rot: {rot}")

        print(f"dial ends at zero {end} times")
        print(f"dial rotated through zero {rot} times")
        print(f"dial clicked at zero {end+rot} total times")
