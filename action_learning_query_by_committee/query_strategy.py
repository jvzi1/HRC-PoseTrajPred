import numpy as np
from scipy.stats import entropy

def vote_entropy(committee_predictions):
    """计算Vote Entropy来度量委员会间的分歧。
    
    Args:
        committee_predictions (list of np.array): 每个委员会成员的预测输出，shape为(N, num_classes)。
    
    Returns:
        entropy_scores (np.array): 每个样本的熵分数。
    """
    votes = np.mean(committee_predictions, axis=0)  # 计算投票平均值
    entropy_scores = entropy(votes.T)  # 计算熵，表示分歧度
    return entropy_scores

def select_samples_by_entropy(committee_predictions, query_size):
    """根据Vote Entropy选择最有信息量的样本。
    
    Args:
        committee_predictions (list of np.array): 委员会成员的预测输出，shape为(N, num_classes)。
        query_size (int): 每次选择的样本数量。
    
    Returns:
        selected_indices (np.array): 被选中样本的索引。
    """
    entropy_scores = vote_entropy(committee_predictions)
    selected_indices = np.argsort(entropy_scores)[-query_size:]  # 选择熵最高的样本
    return selected_indices
