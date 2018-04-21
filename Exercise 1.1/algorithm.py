import random

def partition(array, left, right, part_index):
    part_value = array[part_index]
    array[right], array[part_index] = array[part_index], array[right]
    store_index = left
    
    for i in range(left,right):
        if array[i] < part_value:
            array[i], array[store_index] = array[store_index], array[i]
            store_index += 1
    array[right], array[store_index] = array[store_index], array[right]
    
    return store_index


def quick_select(array, left, right, k):
    while True:
        if left == right:
            return array[left]
            
        part_index = partition(array, left, right, random.randint(left, right))
        #print("After partition:")
        #print(array)

        if k == part_index:
            return array[k]
        elif k < part_index:
            right = part_index-1
        else:
            left = part_index+1
            
def select(array, k):
    return quick_select(array, 0, len(array)-1, k)
    
def quartile(array, n):
    quart_distance = n * (len(array)-1) / 4.0
    quart_index = int(quart_distance)
    quart_value = select(array, quart_index)
    quart_value_next = select(array, quart_index+1)
    
    r = quart_value_next - quart_value
    r = r * (quart_distance - quart_index)
    return quart_value + r
