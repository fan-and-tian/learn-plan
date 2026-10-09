class BankAccount:
    def __init__(self, name, balance=0):
        self.name =  name
        self.balance = balance
    def deposit(self, amount):
        self.balance += amount
        return self.balance
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("取款金额必须大于0")
        if self.balance >= amount:
            self.balance -= amount
        else:
            # 余额不足，抛出异常，交给try捕获
            raise ValueError("余额不足")
        return self.balance
    def show(self):
        print(f"{self.name} 余额: {self.balance}")

# 测试
acc = BankAccount("张三", 1000)
acc.deposit(500)
acc.show()          # 期望: 张三 余额: 1500
acc.withdraw(200)
acc.show()          # 期望: 张三 余额: 1300
try:
    acc.withdraw(9999)
    print("测试失败! 应该抛出异常")
except ValueError:
    print("测试成功! 余额不足时正确抛出异常")
