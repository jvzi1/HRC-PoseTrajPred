import sys
import cv2
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QSlider, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import subprocess
import os
from loguru import logger

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = None
        self.timer = QTimer(self)
        self.isPaused = True
        self.start_frame = None
        self.end_frame = None
        self.total_frames = 0
    
    def initUI(self):
        self.setWindowTitle('人机协作视频动作裁剪')
        self.setGeometry(50, 50, 1600, 700)
        
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        
        self.layout = QVBoxLayout(self.centralWidget)
        
        # QLabel 显示视频
        self.videoLabel = QLabel(self)
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.videoLabel)
        
        # 控制按钮布局
        buttonLayout = QHBoxLayout()
        
        # 加载视频按钮
        self.loadButton = QPushButton('加载视频', self)
        self.loadButton.clicked.connect(self.loadVideo)
        buttonLayout.addWidget(self.loadButton)
        
        # 播放/暂停视频按钮
        self.playButton = QPushButton('播放视频', self)
        self.playButton.clicked.connect(self.playPauseVideo)
        buttonLayout.addWidget(self.playButton)
        
        # 停止视频按钮
        self.stopButton = QPushButton('停止视频', self)
        self.stopButton.clicked.connect(self.stopVideo)
        buttonLayout.addWidget(self.stopButton)
        
        self.layout.addLayout(buttonLayout)
        
        # 视频进度条
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.sliderMoved.connect(self.setPosition)
        self.layout.addWidget(self.slider)
        
        # 裁剪视频按钮
        self.cutButton = QPushButton('裁剪视频', self)
        self.cutButton.clicked.connect(self.cutVideo)
        self.layout.addWidget(self.cutButton)

        # 清除选择按钮
        self.clearButton = QPushButton('清除选择', self)
        self.clearButton.clicked.connect(self.clearSelection)
        self.layout.addWidget(self.clearButton)
    
    def loadVideo(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '', '视频文件 (*.mp4 *.avi *.mkv)')
        if fileName:
            self.cap = cv2.VideoCapture(fileName)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.slider.setMaximum(self.total_frames)
            self.videoPath = fileName
            self.playButton.setText('播放')
            self.start_frame = None
            self.end_frame = None
            logger.info('Video loaded.')
    
    def playPauseVideo(self):
        if self.cap is not None:
            if self.isPaused:
                self.timer.timeout.connect(self.nextFrame)
                self.timer.start(1000 // self.fps)
                self.isPaused = False
                self.playButton.setText('暂停')
            else:
                self.timer.stop()
                self.isPaused = True
                self.playButton.setText('播放')
    
    def stopVideo(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.timer.stop()
            self.isPaused = True
            self.slider.setValue(0)
            self.playButton.setText('播放')
    
    def nextFrame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(frame_pos)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.videoLabel.setPixmap(QPixmap.fromImage(qImg))
        else:
            self.stopVideo()
    
    def setPosition(self, position):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.videoLabel.setPixmap(QPixmap.fromImage(qImg))
    
    def cutVideo(self):
        current_position = self.slider.value()
        if self.start_frame is None:
            self.start_frame = current_position
            logger.info(f"start frame: {self.start_frame}")
        elif self.end_frame is None:
            self.end_frame = current_position
            logger.info(f"end frame: {self.end_frame}")
            self.performCut()
        else:
            logger.warning("已选择起始和结束帧, 出现异常")
    def clearSelection(self):
        self.start_frame = None
        self.end_frame = None
        logger.info("clear selection done")
    
    def performCut(self):
        start_time = self.start_frame / self.fps
        end_time = self.end_frame / self.fps
        
        # 动作标签输入
        categories = ["walk", "carry", "measure", "playphone", "crouch", "operate"]
        category, ok = QInputDialog.getItem(self, '输入动作类别', '请输入动作类别标签:', categories, 0, False)
        if not ok or not category.strip():
            QMessageBox.warning(self, "警告", "标签不能为空！")
            return
        category = category.strip()
        video_path = os.path.dirname(self.videoPath)
        video_name = os.path.basename(self.videoPath)
        video_name = video_name.split(".")[0]
        category_path = os.path.join(video_path, "annotated_videos", category)
        if not os.path.exists(category_path):
            os.makedirs(category_path, exist_ok=True)
        dir_index = 0

        while True:
            sub_dir = os.path.join(category_path, str(dir_index))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            output_path = os.path.join(category_path, str(dir_index), f"{video_name}_{category}_{str(dir_index)}.mp4")
            if not os.path.exists(output_path):
                break
            else:
                dir_index += 1
        
        if output_path:
            command1 = f"ffmpeg -i \"{self.videoPath}\" -ss {start_time} -to {end_time} -c copy \"{output_path}\""
            # fstring中间要使用单引号！！！
            command2 = f"ffmpeg -i \"{self.videoPath.replace('video_1', 'video_2')}\" -ss {start_time} -to {end_time} -c copy \"{output_path.replace('video_1', 'video_2')}\""
            command3 = f"ffmpeg -i \"{self.videoPath.replace('video_1', 'video_3')}\" -ss {start_time} -to {end_time} -c copy \"{output_path.replace('video_1', 'video_3')}\""
            os.system(command1)
            os.system(command2)
            os.system(command3)
            logger.info("视频裁剪完成")
        self.start_frame = None
        self.end_frame = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
