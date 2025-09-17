values = int(input('How many values do you want the Fibonacci sequence to? '))
j = 0
k = 1
final = []
for i in range (values):
    final.append(k)
    l = j + k
    j = k
    k = l


print(final)






