class Hash:
    def __init__(self, max):
        self.max = max
        self.map = {}

    def hash(self, string):
        sentence = string.split()
        number_set = set()
        for word in sentence:
            val = self.hash_string(word)
            self.map[word] = val
            number_set.add(val)
        return number_set

    def hash_string(self, word):
        val = sum(ord(char) for char in word) % self.max
        return val + 1

max_hash = 100000
hash= Hash(max_hash)

string = input("Enter a sentence to hash: ")
number_set = hash.hash(string)

print("Hash values corresponding to each word in the input string:")
print(number_set)
