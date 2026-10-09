def calculator():
    """读入两个数和运算符，输出结果，用异常处理非法输入"""
    pass

# 模拟测试（正常情况）
# 运行时手动输入: 10 + 5 → 期望输出 15
# 运行时手动输入: 10 / 0 → 期望输出 "不能除以零"
# 运行时手动输入: abc + 5 → 期望输出 "输入的不是数字"

# 自动化测试（不依赖input）
def calc(a, op, b):
    """calculator 的非交互版本，用于自动测试"""
    pass

# 测试
assert calc(10, "+", 5) == 15
assert calc(10, "-", 3) == 7
assert calc(4, "*", 5) == 20
assert calc(10, "/", 2) == 5.0
assert calc(10, "/", 0) == "不能除以零"
assert calc("abc", "+", 5) == "输入的不是数字"
print("全部测试成功!")