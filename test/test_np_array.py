import numpy as np



def test1():
    probs = np.random.dirichlet(0.2 * np.ones(0))
    float_probs = np.random.dirichlet(0.2 * np.ones(len(probs)))
    choice_probs = 0.75 * probs + 0.25 * float_probs
    print(probs)
    print(type(probs))
    print(float_probs)
    print(type(float_probs))
    print(choice_probs)
    print(type(choice_probs))

def test2():
    arr = np.zeros([2, 3, 4])
    print(type(arr))
    arr[0][0][0] = 1
    print(arr)
    arr[1][1][1] = 2
    print(arr)

def test3():
    current_players = [1, 2, 1, 2, 1, 2]
    winner = 1
    winner_z = np.zeros(len(current_players))
    if winner != -1:
        winner_z[np.array(current_players) == winner] = 1.0
        winner_z[np.array(current_players) != winner] = -1.0
    print(winner_z)

    acts = [1, 2, 3, 4]
    probs = [5, 6, 7, 8]
    act_probs = zip(acts, probs)
    print(act_probs)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def test4():
    res = np.zeros(2086)
    action_ids = (1, 2)
    visits = (5, 5)
    temp = 1 / 0.2 * np.log(np.array(visits) + 1e-10)
    print(type(temp), temp.shape)
    action_probs = softmax(temp)
    print(type(action_probs), action_probs.shape)
    res[list(action_ids)] = action_probs
    print(res)


if __name__ == '__main__':
    # test1()
    test4()

