
import math
a = int(input('Please enter a value for a: '))
b = int(input('Please enter a value for b: '))
c = int(input('Please enter a value for c: '))

d = math.sqrt(b**2-4*a*c)

a1 = (-b+math.sqrt(b**2-4*a*c))/(2*a)
a2 = (-b-math.sqrt(b**2-4*a*c))/(2*a)

print('The values for x are: ')
print(a1)
print(a2)