class CONFIG:
    # 数据路径
    DATASET_PATH = 'data/videos'
    TRAIN_IMAGES_PATH = 'train'
    VAL_IMAGES_PATH = 'val'
    TEST_IMAGES_PATH = 'test'
    CLIP_LEN = 16  # 每个视频剪辑的帧数

    # 模型参数
    NUM_CLASSES = 10  # 类别数量

    # 训练参数
    NUM_EPOCHS = 10  # 每次迭代的训练轮数
    BATCH_SIZE = 16  # 批大小
    LEARNING_RATE = 1e-4  # 学习率
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 主动学习参数
    INITIAL_TRAIN_SIZE = 0.1  # 初始训练集占比
    COMMITTEE_SIZE = 3  # 委员会成员数量
    QUERY_SIZE = 50  # 每轮主动学习标注的样本数量
