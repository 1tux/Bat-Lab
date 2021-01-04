import random
import random
import numpy as np
from constants import FRAME_RATE 

def get_minimal_shift(minimal_shift_in_minutes = 30):
    return minimal_shift_in_minutes * FRAME_RATE * 60

def shuffling(neuron, num=1000):
    random.seed(1337)
    minimum = get_minimal_shift(10) # 3 * len(neuron) // 10 # 150K - 100 mins -> 45K 30. 45/150 -> 9/30 -> 3/10
    old_idx = set([0])
    idx = 0
    
    idx_margin = 10
    
    for i in range(num):
        while idx // idx_margin in old_idx:
            idx = random.randint(minimum, len(neuron) - minimum)
        old_idx.add(idx // idx_margin)
            
        yield np.roll(neuron, idx) # pd.reindex(concat([neuron[idx:], neuron[:idx]]).reset_index(drop=True)