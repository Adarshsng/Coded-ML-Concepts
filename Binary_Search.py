import numpy as np
print("Let's learn Binary search")

# used for sorted array, acs or desc
# Find the middle element, in asc array  
# if middle element is > target number
# we will search in similar pattern on left hand side



arr = np.array([-4,5,-2,7,4,9,6,10])
arr_srt = np.sort(arr)
target = 10

## approach 1
len = arr_srt.shape[0]
print(arr_srt, len)
while len != 0:
    mid = int(np.floor(len/2))
    
    if target == arr_srt[mid]:
        print('found', np.where(arr==target)[0])
        break
    elif target > arr_srt[mid]:
        arr_srt = arr_srt[mid+1:]
    else:
        arr_srt = arr_srt[:mid]
        
    len = arr_srt.shape[0]
    
## approach 2
arr_srt = -np.sort(-arr)
start = 0
end = arr_srt.shape[0]

if_asc = arr_srt[0] < arr_srt[-1]

print(arr_srt, start, end, if_asc)
while start != end:
    mid = int(start + (end-start)/2)
    if target == arr_srt[mid]:
        print('found', np.where(arr==target)[0])
        break
    elif if_asc:
        if target > arr_srt[mid]:
            start = mid + 1
        else:
            end = mid
    else:
        if target < arr_srt[mid]:
            start = mid + 1
        else:
            end = mid 