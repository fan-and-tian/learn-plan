import time

def fib_recursive(n):
    """递归实现斐波那契"""
    if n <=1:
        return n
    else:
        return fib_recursive(n-1)+fib_recursive(n-2)   

cache = {}
def fib_cached(n):
    """带字典缓存的斐波那契"""
    if n in cache:
        return cache[n]
    if n <= 1:
        cache[n]= n
        return n   
    else:
        cache[n] = fib_cached(n-1) + fib_cached(n-2)
        return cache[n] 
# 测试正确性
assert fib_recursive(0) == 0
assert fib_recursive(1) == 1
assert fib_recursive(10) == 55
assert fib_cached(0) == 0
assert fib_cached(10) == 55
assert fib_cached(30) == 832040
print("正确性测试通过!")

# 测试速度
start = time.time()
fib_cached(35)
cached_time = time.time() - start

start = time.time()
fib_recursive(30)
recursive_time = time.time() - start

print(f"递归 fib(30): {recursive_time:.2f}s")
print(f"缓存 fib(35): {cached_time:.4f}s")
if cached_time < recursive_time:
    print("测试成功! 缓存版远快于递归版")
else:
    print("测试失败! 缓存版应该更快")