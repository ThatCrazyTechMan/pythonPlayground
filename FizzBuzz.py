a = 5
b = 3
var1 = str(input('What should multiples of 3 return: '))

var2 = str(input('What should multiples of 5 return: '))
start = int(input('Where should it start? '))
end = int(input('Where should it end? '))


for i in range(start, end+1):
    if i % b == 0 and i % a == 0:
        print(f"{var1}{var2}")
    elif i % b == 0:
        print(var1)
    elif i % a == 0:
        print(var2)
    else:
        print(i)

