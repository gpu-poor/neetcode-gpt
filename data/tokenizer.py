from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        merges = []
        corpus = list(corpus)
        iteration = 0
        while iteration < num_merges and len(corpus) > 1:
            pairs = []
            #pair_count_map = Counter()
            for idx in range(len(corpus)-1):
                pairs.append((corpus[idx], corpus[idx+1]))
            sorted_common_pairs = Counter(pairs).most_common()
            max_count = sorted_common_pairs[0][1]
            tie = []
            for item in sorted_common_pairs:
                if item[1] == max_count:
                    tie.append(item[0])
            tie.sort()
            merges.append(tie[0])

            # change corpus
            new_corpus = []
            idx = 0
            while idx <= len(corpus)-2: 
                pair = (corpus[idx], corpus[idx+1])
                if pair == merges[-1]:
                    new_corpus.append("".join(pair))
                    idx+=2
                else:
                    new_corpus.append(corpus[idx])
                    idx+=1

            corpus = new_corpus
            iteration+=1


        return merges
        pass
