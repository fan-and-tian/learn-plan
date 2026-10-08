import os

def batch_rename(folder):
    """把folder里所有.jpg文件重命名为001.jpg, 002.jpg..."""
    jpg_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    jpg_files.sort()
    for i, filename in enumerate(jpg_files, start=1):
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, f"{i:03d}.jpg")
        os.rename(old_path, new_path)

# 先创建测试文件夹和假文件
os.makedirs("test_rename", exist_ok=True)
for name in ["photo_a.jpg", "photo_b.jpg", "photo_c.jpg", "readme.txt"]:
    open(f"test_rename/{name}", "w").close()

batch_rename("test_rename")

# 测试
files = sorted(os.listdir("test_rename"))
print("重命名后:", files)
if files == ["001.jpg", "002.jpg", "003.jpg", "readme.txt"]:
    print("测试成功!")
else:
    print("测试失败! 期望 001.jpg 002.jpg 003.jpg readme.txt")