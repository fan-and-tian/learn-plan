def f(x):
    return x*x
A = []
for a in range(2,101,2):
    A.append(a)
l = map(f, A)
print(list(l))