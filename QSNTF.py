# QuadraticSequenceNthTermFinder.py


import numpy as np

first4str = str(input('Enter the first 4 numbers in the sequence: '))
first4arr = [int(num.strip()) for num in first4str.split(',')]



print("First 4 values: ", first4arr)
firstDiff = [first4arr[i+1] - first4arr[i] for i in range(len(first4arr) - 1)]
print("First differences: ", firstDiff)

secondDiff = [firstDiff[i+1] - firstDiff[i] for i in range(len(firstDiff) - 1)]
print("Second Differences", secondDiff)

secondDiffVal = secondDiff[0]
A = secondDiffVal / 2
print("A = ", A)


length = len(first4arr)
lengthArray = list(range(1, length + 1))
lenArrSquared = [A*x**2 for x in lengthArray]


first4arr_np = np.array(first4arr)
lenArrSquared_np = np.array(lenArrSquared)


bnc = first4arr_np - lenArrSquared_np
bnc = bnc.tolist()
print("Bn+C = ", bnc)

bncFirstDiff = [bnc[i+1] - bnc[i] for i in range(len(bnc) - 1)]
bncFirstDiff = bncFirstDiff[0]
print("The first difference of sequence Bn+C = ", bncFirstDiff)


print(f"So far, the equation is:",A,"n^2+",bncFirstDiff,"+c")

bnc_np = np.array(bnc)
bncFirstDiff_np = np.array(bncFirstDiff)
C = bnc_np - bncFirstDiff_np
C = C.tolist()
C = C[0]



print(f"The equation is:",A,"n^2+",bncFirstDiff,"n","+",C)