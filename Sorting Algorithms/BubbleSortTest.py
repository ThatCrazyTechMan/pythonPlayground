array = [1, 3, 6, 9, 7, 4, 2, 5]
n = len(array)

for i in range(n):
    for j in range(n - i - 1):
        if array[j] > array[j + 1]:
            array[j], array[j + 1] = array[j + 1], array[j]

print(array)
