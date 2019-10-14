def a(a):
    def _b(k):
        return 4

    return _b

x = a(9)
print(x(564))