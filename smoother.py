import numpy as np
class ActionSegmentSmoother:
    def __init__(self, alpha=0.9, min_segment_length=16, delay_threshold=5):
        self.alpha = alpha
        self.min_segment_length = min_segment_length
        self.delay_threshold = delay_threshold
        self.current_label = None
        self.segment_start = 0
        self.delay_counter = 0
        self.smoothed_label_probs = None

    def smooth_and_segment(self, frame_probs, class_names):
        segments = []
        
        for frame_count, probs in enumerate(frame_probs):

            # 设置动态的alpha，根据最高置信度来更改alpha值
            confidence = np.max(probs)
            alpha = self.alpha * confidence
            # 加权平滑
            if self.smoothed_label_probs is None:
                self.smoothed_label_probs = probs
            else:
                self.smoothed_label_probs = (
                    alpha * probs + (1 - alpha) * self.smoothed_label_probs
                )

            # 获取平滑后的标签
            label_num = np.argmax(self.smoothed_label_probs)
            label = class_names[label_num]

            # 延迟确认和最小片段长度检查
            if self.current_label is None:
                # 初始化标签和片段开始帧
                self.current_label = label
                self.segment_start = frame_count
            elif label != self.current_label:
                # 检测到标签变化，启动延迟确认
                self.delay_counter += 1
                if self.delay_counter >= self.delay_threshold:
                    # 延迟确认通过，检查片段长度
                    segment_length = frame_count - self.segment_start
                    if segment_length >= self.min_segment_length:
                        # 符合最小片段长度，记录该片段
                        segments.append((self.segment_start, frame_count - 1, self.current_label))
                    # 更新当前标签和片段开始帧
                    self.current_label = label
                    self.segment_start = frame_count
                    self.delay_counter = 0
            else:
                # 标签未变化，重置延迟计数
                self.delay_counter = 0

        # 处理最后一个片段
        if self.current_label is not None:
            segment_length = len(frame_probs) - self.segment_start
            if segment_length >= self.min_segment_length:
                segments.append((self.segment_start, len(frame_probs) - 1, self.current_label))

        return segments
