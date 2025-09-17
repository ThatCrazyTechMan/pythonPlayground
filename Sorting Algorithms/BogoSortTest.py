import random

array = [1, 5, 3, 6, 3, 9, 0, 3, 2, 8, 6]


def is_sorted(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


while not is_sorted(array):
    random.shuffle(array)
    print(array)
print(array)
