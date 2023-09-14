import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Game Experiment.")
    parser.add_argument("--agent_num", type=int, help="agent num to resolve")
    args, remaining_args = parser.parse_known_args()
    print(args)
    remaining_args_num = len(remaining_args)
    print(remaining_args_num)
    for i in range(remaining_args_num):
        parser.add_argument("--input" + str(i), type=str,
                            help="input of agent")
    args, remaining_args = parser.parse_known_args()
    print(args, remaining_args)
