def count_words(text):
    words = text.split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

text = "python is fun and python is powerful"
print(count_words(text))
# 期望输出：{'python': 2, 'is': 2, 'fun': 1, 'and': 1, 'powerful': 1}