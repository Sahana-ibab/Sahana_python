# Implement ordinal encoding and one-hot encoding methods in Python from scratch.

# oneHot Encoding

class Ordinal_encoding():
    def fit(self, x):
        self.classes= sorted(set(x))
        self.encoding = {cat: i+1 for i, cat in enumerate(self.classes) }
        return self.encoding

    def transform(self, x):
        return [self.encoding[i] for i in x]

def main():
    x = ["blue", "green", "red", "blue", "red"]
    oe = Ordinal_encoding()
    print(oe.fit(x))
    print(oe.transform(x))

if __name__ == '__main__':
    main()
#
# class OneHotEncoder:
#     def fit(self, X):
#         self.classes = sorted(set(X))
#         self.encoding = {cat: [0] * i + [1] + [0] * (len(self.classes) - (i + 1)) for i, cat in enumerate(self.classes)}
#         return self.encoding
#
#     def transform(self, X):
#         return [self.encoding[x] for x in X]
#
# def main():
#     X = ['green', 'red', 'blue', 'blue', 'red']
#     encoder = OneHotEncoder()
#     print("One-Hot Mapping:", encoder.fit(X))
#     print("Transformed Data:", encoder.transform(X))
#
# if __name__ == '__main__':
#     main()
