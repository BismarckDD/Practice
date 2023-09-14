import sys
sys.path.append("..")
import os
import pickle
import time
from collections import deque
import zip_array

file_path = "./train_data_buffer.pkl"

def check_data():
    if os.path.exists(file_path):
        with open(file_path, "rb") as file_handle:
            data_dict = pickle.load(file_handle)
            data_buffer = deque()
            data_buffer.extend(data_dict["data_buffer"])
            iters = data_dict["iters"]
            del data_dict  # is necessary?
            print(len(data_buffer), iters)
            # for item in data_buffer:
            #     state, action_probs, winner = zip_array.recovery_state_mcts_prob(item)
            #     print(state)
            #     time.sleep(1)

if __name__ == "__main__":
    check_data()
