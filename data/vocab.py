from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = list(text)
        chars.sort()
        stoi = dict()
        itos = dict()
        idx = 0    
        for char in chars:
            if char not in stoi:
                stoi[char] = idx
                itos[idx] = char
                idx+=1
        return stoi, itos
        

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoded_list = []
        for char in text:
            encoded_list.append(stoi[char])
        return encoded_list
        

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded_list = []
        for id in ids:
            decoded_list.append(itos[id])
        return "".join(decoded_list)
