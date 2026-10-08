def is_odd(n):
    return lambda n: n % 2 == 1

L = list(filter(is_odd(1), range(1, 20)))

print(L)
