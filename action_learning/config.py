import torch

class CONFIG:
    # 固定种子
    SEED = 41

    # 数据路径
    DATASET_PATH = 'rec_728_frame'
    TRAIN_IMAGES_PATH = 'train'
    VAL_IMAGES_PATH = 'val'
    TEST_IMAGES_PATH = 'test'

    # 模型参数
    NUM_CLASSES = 6  # 动作类别数量
    CLIP_LEN = 16  # 视频帧的长度

    # 训练参数
    TRAIN_SIZE_RATIO = 0.1  # 初始训练集的比例（10%）
    NUM_EPOCHS = 100  # 训练轮数
    LEARNING_RATE = 1e-4  # 学习率
    BATCH_SIZE = 16  # 批量大小

    # 设备设置
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 保存路径
    SAVE_DIR = 'model_result'

    # 迭代设置
    ITER_TIMES = 5