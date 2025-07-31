# Implement ordinal encoding and one-hot encoding methods in Python from scratch.

# class ordinal_encoder():
#     def __init__(self, X):
#         self.X=X
#
#     def fit(self):
#         self.sets= list(sorted(set(self.X)))
#         self.dict_sets={}
#         for i in range(0, len(self.sets)):
#             self.dict_sets[self.sets[i]]=i
#         return self.dict_sets
#
#     def transform(self, X):
#         self.X=X
#         for i in range(0,len(self.X)):
#             self.X[i] = self.dict_sets[self.X[i]]
#         return self.X
#
# def main():
#     X=['green', 'red', 'blue', 'blue', 'red']
#     ord = ordinal_encoder(X)
#     result = ord.fit()
#     print(result)
#     result = ord.transform(X)
#     print(result)
#
# if __name__ == '__main__':
#     main()

class OneHotEncoder:
    def fit(self, X):
        self.classes = sorted(set(X))
        self.encoding = {cat: [0] * i + [1] + [0] * (len(self.classes) - i - 1) for i, cat in enumerate(self.classes)}
        return self.encoding

    def transform(self, X):
        return [self.encoding[x] for x in X]

def main():
    X = ['green', 'red', 'blue', 'blue', 'red']
    encoder = OneHotEncoder()
    print("One-Hot Mapping:", encoder.fit(X))
    print("Transformed Data:", encoder.transform(X))

if __name__ == '__main__':
    main()
