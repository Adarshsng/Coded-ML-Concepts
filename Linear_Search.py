import numpy as np
print('Leats learn Linear search')

arr = np.array([1,8,9,3,4,7])
arr = np.array([[1,8,9,3,4,7],[1,8,9,3,4,7]])

# find if 14 exist or not in arr
if 14 in arr:
    print('!4 does exists')
else:
    print('!4 does not exists')
# find if 7 exist or not in arr
print("Does 7 exists in arr ?","True" if 7 in arr else "False")

# lets do it without fancy functions !!
def search_arr(arr, num):
    arr = arr.ravel()
    if len(arr) == 0:
        return('array length is 0')
    for j,i in enumerate(arr) :
        if num == i:
            return(j)
    return -1
print(search_arr(arr,7))

def search_string(arr, num):
    arr = arr.ravel()
    # arr = sum([list(i) for i in arr],[])
    arr = np.hstack([list(i) for i in arr])
    return search_arr(arr[2:6],num)
    
print(search_string(np.array(['Adarsh','Singh']),'a'))

# find min
print(min(arr.ravel()))
def find_min(arr):
    arr = arr.ravel()
    min = arr[0]
    for i in arr:
        if i < min:
            min = i
    return min
print(find_min(arr))

nums = np.array([12,345,2,6,7896])
 
for i in nums:
    if len(list(str(i))) % 2 ==0:
        print(i)
        
for i in nums:
    j = i
    count = 0
    while i>0:
        i /= 10
        count += 1
    if count%2==0:
        print(j)
        
rich_arr = np.array([[1,3],[4,5],[8,7]])
print(max([sum(i) for i in rich_arr]))