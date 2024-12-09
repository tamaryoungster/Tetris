import numpy as np
board = np.zeros([10,10])

piece = np.array([[1,1,1],[0,0,1]])
print(piece)
print(np.rot90(piece, k=3))
print(piece)


# temp_Board = change board to 0 or 1
# np.sum by rows -> array of sums
# find row that is 10
#if row = 10 -> to delete