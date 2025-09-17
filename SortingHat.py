# Write code below 💖
G = 0
R = 0
H = 0
S = 0

q1 = int(input('Q1 Do you like Dawn or Dusk?, 1 Dawn, 2 Dusk'))
if q1 == 1:
    G = G + 1
    R = R + 1
elif q1 == 2:
    H = H + 1
    S = S + 1
else:
    print('Invalid input')

q2 = int(
    input('Q2 When I’m dead, I want people to remember me as: 1: The Good, 2: The Great, 3: The Wise, 4: The Bold'))
if q2 == 1:
    H = H + 2
elif q2 == 2:
    S = S + 2
elif q2 == 3:
    R = R + 2
elif q2 == 4:
    G = G + 2
else:
    print('Invalid input')

q3 = int(input(
    'Q3 Which kind of instrument most pleases your ear? 1: The violin, 2: The trumpet, 3: The piano, 4: The drum'))
if q3 == 1:
    S = (S + 4)
elif q3 == 2:
    H = (H + 4)
elif q3 == 3:
    R = (R + 4)
elif q3 == 4:
    G = (G + 4)
else:
    print('Invalid input')

print(f'Gryffindor: {G}')

print(f'Ravenclaw: {R}')

print(f'Hufflepuff: {H}')

print(f'Slytherin: {S}')


allScores = [q1, q2, q3]
sorted(allScores)

finalScore = allScores[-1]

if finalScore == G:
    print('Your house is Gryffindor')
elif finalScore == H:
    print('Your house is Hufflepuff')
elif finalScore == R:
    print('Your house is Ravenclaw')
elif finalScore == S:
    print('Your house is Slytherin')
