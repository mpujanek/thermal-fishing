for i in range(10000000):
    if i == sum(map(int, str(i**2))):
        print(i)
