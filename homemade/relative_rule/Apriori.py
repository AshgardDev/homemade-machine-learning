from itertools import combinations

import numpy as np

class Apriori:
    def __init__(self, data, min_support = 0.5):
        self.data = data
        self.min_support = min_support
        self.data_one_hot, self.item_index_dict = self.to_one_hot(data)

        result = []
        candidate_items = list(self.item_index_dict.keys())
        candidate_items.sort()
        while len(candidate_items) >= 1:
            frequent_item_dict, frequent_item_keys = self.scan(candidate_items) #频繁项集
            if frequent_item_dict:
                result.append(frequent_item_dict)
                candidate_items = self.get_candidate_items(frequent_item_keys)  # 候选集

        support = {}
        for res_dict in result:
            for k, v in res_dict.items():
                support[k] = v
        self.support = support

    def get_candidate_items(self, frequent_item_keys):
        candidate_items = []
        if len(frequent_item_keys) > 1:
            frequent_items_len = len(frequent_item_keys)
            for i in range(frequent_items_len):
                for j in range(i + 1, frequent_items_len):
                    if self.hava_same_pre_candidate_key(frequent_item_keys[i], frequent_item_keys[j]):
                        candidate_items.append(frequent_item_keys[i] + "," + frequent_item_keys[j].split(",")[-1])
        return candidate_items

    def hava_same_pre_candidate_key(self, frequent_items_prev, frequent_items_next):
        prefix_prev = ",".join(frequent_items_prev.split(",")[:-1])
        prefix_next = ",".join(frequent_items_next.split(",")[:-1])
        return prefix_prev == prefix_next

    def scan(self, candidate_items):
        freq_item_dict = {}
        freq_item_keys = []
        for item in candidate_items:
            indexs = self.to_index(item)
            count = 0
            for row in self.data_one_hot:
                if np.sum(row[indexs]) == len(indexs):
                    count += 1
            support = count / len(self.data_one_hot)
            if support >= self.min_support:
                freq_item_dict[item] = support
                freq_item_keys.append(item)
        return freq_item_dict, freq_item_keys

    def to_index(self, item):
        indexs = []
        keys = item.split(",")
        for key in keys:
            indexs.append(self.item_index_dict[key.strip()])
        return indexs

    def to_one_hot(self, data):
        item_type = set()
        for items in data:
            item_type.update(items)
        item_type = list(item_type)
        item_type.sort()
        item_index_dict = {item: i for i, item in enumerate(item_type)}
        one_hot = np.zeros((len(data), len(item_type)), dtype=int)
        for idx, items in enumerate(data):
            for item in items:
                one_hot[idx, item_index_dict[item]] = 1
        return one_hot, item_index_dict

    def generate_rules(self, freq_itemsets, min_conf=0.6):
        """
        freq_itemsets: DataFrame, 包含 [support, itemsets]
        min_conf: 最小置信度
        """
        rules = []
        for itemset, supp in freq_itemsets.items():
            itemset_arr = itemset.split(",")
            sorted(itemset_arr)
            if len(itemset_arr) < 2:
                continue
            # 遍历所有可能的划分
            for i in range(1, len(itemset_arr)):
                for left in combinations(itemset_arr, i):
                    left = sorted(left)
                    right = sorted(set(itemset_arr) - set(left))
                    supp_xy = freq_itemsets[itemset]
                    supp_x = freq_itemsets[",".join(left)]
                    supp_y = freq_itemsets[",".join(right)]
                    conf = supp_xy / supp_x
                    lift = conf / supp_y
                    if conf >= min_conf:
                        rules.append({
                            "X": left,
                            "Y": right,
                            "support": supp_xy,
                            "confidence": conf,
                            "lift": lift
                        })
        return rules

if __name__ == '__main__':
    data = {
        "Tid": [1, 2, 3, 4],
        "Items" : [
            ["A", "C", "D"],
            ["B", "C", "E"],
            ["A", "B", "C", "E"],
            ["B", "E"],
        ]
    }
    apriori = Apriori(data["Items"], min_support = 0.5)
    print(apriori.support)
    print(apriori.generate_rules(apriori.support, 1))
