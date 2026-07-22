# Python program to count all possible paths
# from top left to bottom right

# Returns count of possible paths to reach cell
# at row number m and column number n from the
# topmost leftmost cell (cell at 1, 1)
def numberOfPaths(m, n):
    # Create a 2D table to store
    # results of subproblems
    # one-liner logic to take input for rows and columns
    # mat = [[int(input()) for x in range (C)] for y in range(R)]

    count = [[0 for x in range(n)] for y in range(m)]
    count[0][0] = 1
    # Calculate count of paths for other
    # cells in bottom-up
    # manner using the recursive solution
    for i in range(1, m):
        for j in range(1, n):
            if i == 1 and j >= 2:
                count[i][j] = count[i - 1][j - 2]
            elif j == 1 and i >= 2:
                count[i][j] = count[i - 2][j - 1]
            else:
                count[i][j] = count[i - 1][j - 2] + count[i - 2][j - 1]
    return count[m - 1][n - 1]

# Driver program to test above function
n = 22
print(numberOfPaths(n, n))

# This code is contributed by Aditi Sharma
