n = int(input())
nails = list(map(int, input().split()))

def solve(nails: list[int]):
    nails.sort()
    res, i = nails[1] - nails[0], 1
    while i < len(nails) - 1:
        if i + 2 >= len(nails) or (not i + 2 == len(nails) - 1 and nails[i + 1] - nails[i] < nails[i + 2] - nails[i + 1]):
            res += nails[i + 1] - nails[i]
            i += 1
        else:
            res += nails[i + 2] - nails[i + 1]
            i += 2
    return res

print(solve(nails))