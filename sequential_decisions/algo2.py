def knapSack(pop, val, Z,n):
    K = [[999999 for x in range(Z+1)] for x in range(n)]
    for i in range(n):
        for b in range(Z+1): # must go to Z+1 so that b can be 12
            if val[i] >= b:
                K[i][b] = min(K[i - 1][b], pop[i])
            elif val[i] < b:
                K[i][b] = min(K[i - 1][b], pop[i] + K[i-1][b-val[i]])
        print(K[i])
    return K[n-1][Z]


# Driver code
pop = [200, 100, 30, 700, 250]
val = [5, 2, 2, 7, 6]
Z = 300
n = len(val)
print('Answer: ', knapSack(pop, val, Z,n))
