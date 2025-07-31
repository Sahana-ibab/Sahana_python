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