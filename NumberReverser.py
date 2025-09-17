num = int(input('Enter the number: '))
digits = [int(digit) for digit in str(num)]


digitsReversed = list(reversed(digits))
print(digitsReversed)