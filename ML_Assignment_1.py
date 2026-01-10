import random
def pairs(lst):
    count = 0
    n = len(lst)
    for i in range(n):
        for j in range(i+1, n):
            if lst[i] + lst[j] == 10:
                count += 1
    return count

def find(lst):
    if len(lst) < 3:
        return "Range determination not possible"

    minimum = lst[0]
    maximum = lst[0]

    for i in lst:
        if i < minimum:
            minimum = i
        if i > maximum:
            maximum = i

    return maximum - minimum


def mult_the_matrix(A, B):
    result = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                result[i][j] += A[i][k] * B[k][j]
    return result


def power(A, m):
    result = A
    for _ in range(m-1):
        result = mult_the_matrix(result, A)
    return result


def repeats(s):
    char = ''
    mcount = 0

    for ch in s:
        if ch.isalpha():     
            count = 0
            for c in s:         
                if ch == c:
                    count = count + 1

            if count > mcount:
                mcount = count
                char = ch

    return char, mcount


def three_m(arr): 
    n = len(arr)
    total = 0
    for i in arr:
        total += i
    mean = total / n
    a = sorted(arr)
    if n % 2 == 1:
        median = a[n // 2]
    else:
        median = (a[n // 2 - 1] + a[n // 2]) / 2

    freq = {}
    for x in arr:
        freq[x] = freq.get(x, 0) + 1

    max_count = 0
    mode = None
    for k in freq:
        if freq[k] > max_count:
            max_count = freq[k]
            mode = k

    return mean, median, mode



lst1 = [2, 7, 4, 1, 3, 6]
print("Pairs with sum 10:", pairs(lst1))

lst2 = [5, 3, 8, 1, 0, 4]
print("Range:", find(lst2))

A = [[1, 2],
     [3, 4]]
m = 4
print("A^m:", power(A, m))

s = "dracula"
print(repeats(s))

arr = [random.randint(1, 100) for _ in range(25)]
print(three_m(arr))
