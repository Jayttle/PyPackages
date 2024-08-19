def count_and_print_reverse_pairs(arr):
    def merge_sort(arr, temp_arr, left, right):
        if left >= right:
            return 0
        mid = left + (right - left) // 2
        
        count = merge_sort(arr, temp_arr, left, mid) + merge_sort(arr, temp_arr, mid + 1, right)
        count += merge_and_count(arr, temp_arr, left, mid, right)
        
        return count
    
    def merge_and_count(arr, temp_arr, left, mid, right):
        i = left
        j = mid + 1
        k = left
        count = 0
        
        # Merge arrays while counting inversions
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                # Print inversions
                for l in range(i, mid + 1):
                    print(f'逆序对: ({arr[l]}, {arr[j]})')
                
                temp_arr[k] = arr[j]
                count += (mid - i + 1)
                j += 1
            k += 1
        
        # Copy the remaining elements of left subarray
        while i <= mid:
            temp_arr[k] = arr[i]
            i += 1
            k += 1
        
        # Copy the remaining elements of right subarray
        while j <= right:
            temp_arr[k] = arr[j]
            j += 1
            k += 1
        
        # Copy the sorted subarray into Original array
        for i in range(left, right + 1):
            arr[i] = temp_arr[i]
        
        return count

    # Create a temporary array
    temp_arr = [0] * len(arr)
    total_count = merge_sort(arr, temp_arr, 0, len(arr) - 1)
    return total_count

if __name__ == '__main__':
    arr = [4, 3, 5, 7, 1, 6]
    count = count_and_print_reverse_pairs(arr)
    print(f'总逆序对数 = {count}')
