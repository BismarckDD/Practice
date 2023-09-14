

TRAIN_DATA_BUFFER_PATH = "train_data_buffer_path"
MODEL_PATH = "model_path"
DATA_BUFFER = "data_buffer"
ITERS = "iters"

CONFIG = {
    "use_redis": False,
    "kill_action": 200,  # 和棋回合数
    "dirichlet": 0.2,  # 中国象棋:0.2;国际象棋:0.3;日本将棋:0.15;围棋:0.03.
    "train_data_buffer_path": "train_data_buffer.pkl",
    "model_path": "current_model.pkl",
    "train_update_interval": 600,
    "c_puct": 5,  # factor of ucb.
    "temperature": 1e-3,
    "play_out": 1000,
    "buffer_size": 81920,  # 训练数据中的 (state, action_probs, winner) 元组的数量
    "batch_size": 1024,
    "epoch": 5,
    "kl_targ": 0.02,
    "game_batch_num": 2000,
    "check_freq": 500,
}

TOTAL_AVAILABLE_ACTION_NUM = 2086
