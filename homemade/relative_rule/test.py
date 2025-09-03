def get_all_sub_seq(items, r):
    length = len(items)
    if r > length:
        return

    sub_seqs = []
    indices = list(range(r)) # 0 1
    sub_seqs.append(get_sub(items, indices))
    while True:
        pos = compute_pos(items, indices) ## 计算可推进位置
        if pos is None:
            break
        ## 更新下标
        indices[pos] = indices[pos] + 1
        while pos < (r - 1):
            indices[pos + 1] = indices[pos] + 1
            pos += 1
        sub_seqs.append(get_sub(items, indices))
    return sub_seqs

def compute_pos(items, indices):
    length = len(items)
    indices_len = len(indices)
    for i in range(indices_len):
        if indices[indices_len -i - 1] < (length-i-1):
            return indices_len -i - 1
    return None

def get_sub(items, indices):
    return tuple([items[i] for i in indices])


print(get_all_sub_seq(['A', 'B', 'C', 'D', 'E'], 3))