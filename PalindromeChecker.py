word = str(input('Enter the word: '))

char_list = list(word)
charString = "".join(char_list)


reverse = list(reversed(char_list))
reverseString = "".join(reverse)

if reverseString == charString:
    print('This is a palindrome!')
else:
    print('This is not a palindrome!')





