import numpy as np
import time


def test():

    # zip 和 map 返回的都是一个一次性迭代器
    legal_actions = [1, 31, 93, 2001, 2081]
    legal_action_ids = map(lambda e: e, legal_actions)
    # print(type(legal_action_ids))
    # print(len(list(legal_action_ids)))
    # print(len(list(legal_action_ids)))
    act_probs = np.ones(200000)
    for i in range(2086):
        act_probs[i] = float(i)
    print(len(act_probs))
    print(len(act_probs))
    res = zip(legal_action_ids, map(lambda key: act_probs[key], legal_action_ids))
    print(type(res))
    for k, v in res:
        print(type(k), type(v), k, v)
    print("--------")
    for k, v in res:
        print(type(k), type(v), k, v)

if __name__ == "__main__":
    test()
    print(time.time())
