import numpy as np

path = "./residual_mask.npy"

arr = np.load(path)

print(len(arr))

#print(arr)

arr1 = np.logical_not(arr).astype(np.uint8)

print("arr1:", arr1)

#where = np.where(arr == 1)

#print(where)

path2 = "./linker_mask.npy"

arr2 = np.load(path2)

print("Linker mask:", arr2)

xor_arr = np.bitwise_xor(arr1, arr2)
print("XOR array:", xor_arr)
print("Indices where XOR array is 1:", np.where(xor_arr == 1))

