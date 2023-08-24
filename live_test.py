#pyuic5 -o D:\App\Tracking\GUI\live_ui.py D:\App\Tracking\GUI\live.ui
#  D:/CC/APP/GUI/gui.ui

import sys
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
import numpy as np

from track_ID import track_go
from demo import *
from yolo_v3 import YOLO3
from yolo_v4 import YOLO4

from GUI.live_ui import Ui_MainWindow

from PyQt5.QtCore import *

import cv2
import os

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.myPath='D:/App/Tracking/'
        self.downloadPath="C:/Users/zhouc/Downloads/"

        #Video Setups
        self.videoPath=self.myPath+"videos/init/outputrd.mp4"
        self.videoPath2=PyQt5.QtCore.QUrl('file:///D:/App/Tracking/videos/init/outputrd.mp4')
        self.mVideoWin = QVideoWidget(self)
        self.mVideoWin.setGeometry(142,31,640,480)
        self.mVideoWin.hide()
        self.player = QMediaPlayer(self)	
        self.player.setVideoOutput(self.mVideoWin)
        self.playButton.clicked.connect(self.player.play)
        self.stopButton.clicked.connect(self.player.pause)
        # self.player.durationChanged.connect(self.MediaTime)
        self.timer = QTimer()
        self.maxValue = 1000
        self.videoSlider.sliderMoved.connect(self.slider_progress_moved)
        self.videoSlider.sliderReleased.connect(self.slider_progress_released)
        self.videoTrackButton.clicked.connect(self.track_video)


        #Camera Setups
        pixmap = QtGui.QPixmap(self.myPath+"GUI/open_camera.jpg")
        scaled_pixmap = pixmap.scaled(self.c1Label.width(), self.c1Label.height(), Qt.KeepAspectRatio)
        self.c1Label.setPixmap(scaled_pixmap)
        self.openCameraButton.clicked.connect(self.open_camera)
        self.closeCameraButton.setCheckable(True)


        #Live Setups
        self.liveButton.clicked.connect(self.track_live)
        self.liveButton.setCheckable(True)
        self.liveButton.setChecked(False)

        #Page Buttons Setups
        self.c1Button.clicked.connect(self.display_page1)
        self.c2Button.clicked.connect(self.display_page2)
        self.c3Button.clicked.connect(self.display_page3)
        self.c4Button.clicked.connect(self.display_page4)


        #Menu Actions Setups
        self.actionPicture.triggered.connect(self.open_image)
        self.actionVideo.triggered.connect(self.open_video)
        self.actionRecord.triggered.connect(self.export_record)


    def display_page1(self): 
        self.mVideoWin.hide()
        self.stackedWidget.setCurrentIndex(0)

    def open_camera(self):
        camera_ID = 0
        cameraCapture = cv2.VideoCapture(camera_ID)
        success, frame = cameraCapture.read()
        width = frame.shape[1]
        height = frame.shape[0]
        camera_path = self.myPath+"videos/init/record.avi"

        if not os.path.isfile(camera_path):
            raise FileExistsError
        track_camera_out = cv2.VideoWriter(camera_path, cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (width,height))
        while self.radioButton.isChecked():
            if cv2.waitKey(1) == 27:
                break
            if self.closeCameraButton.isChecked():
                break
            success, frame = cameraCapture.read()
            track_camera_out.write(frame)
            print("Frame write")
            cv2.waitKey(1)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label_data = QtGui.QImage(img.data, width, height, width*3, QtGui.QImage.Format_RGB888)  
            self.c1Label.setPixmap(QtGui.QPixmap.fromImage(label_data))
            
        while not self.radioButton.isChecked():
            if cv2.waitKey(1) == 27:
                break
            success, frame = cameraCapture.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label_data = QtGui.QImage(img.data, width, height, width*3, QtGui.QImage.Format_RGB888)  
            self.c1Label.setPixmap(QtGui.QPixmap.fromImage(label_data))
            if self.closeCameraButton.isChecked():
                break
        
        self.closeCameraButton.setChecked(False)
        cameraCapture.release()
        track_camera_out.release()
        self.c1Label.clear()  
        pixmap = QtGui.QPixmap(self.myPath+"GUI/open_camera.jpg")
        scaled_pixmap = pixmap.scaled(self.c1Label.width(), self.c1Label.height(), Qt.KeepAspectRatio)
        self.c1Label.setPixmap(scaled_pixmap)

    def export_record(self):
        import shutil
        source_path = self.myPath+"videos/init/record.avi"
        shutil.copy(source_path, self.downloadPath)
        QMessageBox.information(self,"","Video downloaded")


    def display_page2(self):
        self.mVideoWin.hide()
        self.stackedWidget.setCurrentIndex(1)
        self.c2Label.setPixmap(QtGui.QPixmap(self.myPath+"inputs/input_picture.jpg").scaled(self.c2Label.width(), self.c2Label.height(), Qt.KeepAspectRatio))

    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "Select Picture", "", "*.jpg;;*.PNG;;All Files(*)")
        jpg = QtGui.QPixmap(imgName)
        jpg.save(self.myPath+"inputs/input_picture.jpg")
        # jpg = QtGui.QPixmap(imgName).scaled(self.c2Label.width(), self.c2Label.height())
        self.c2Label.setPixmap(jpg.scaled(self.c2Label.width(), self.c2Label.height(), Qt.KeepAspectRatio))


    def display_page3(self):
        self.stackedWidget.setCurrentIndex(2)
        self.mVideoWin.show()
        self.video_setup()
        self.player.play()

    def track_video(self):
        args.videos=[self.videoPath]
        main(YOLO4())

    def open_video(self):
        videoPath = QFileDialog.getOpenFileUrl(self, "Select File")[0]
        self.videoPath = str(videoPath)[27:len(str(videoPath))-2]
        print(videoPath)
        self.videoPath2 = videoPath
        self.display_page3()

    def video_setup(self):
        self.player.setMedia(QMediaContent(self.videoPath2))
        self.videoSlider.setEnabled(True)
        self.videoSlider.setRange(0,self.maxValue)
        self.FLAG_PLAY = True
        self.timer.setInterval(1000)
        self.timer.start()
        self.timer.timeout.connect(self.onTimerOut)

    def onTimerOut(self):
        self.videoSlider.setValue(round(self.player.position()*self.maxValue/self.player.duration()))
        m, s = divmod(self.player.position() / 1000, 60)
        h, m = divmod(m, 60)
        self.slider_label.setText("%02d:%02d:%02d" % (h, m, s))
    
    def slider_progress_moved(self):
        self.player.pause()
        self.timer.stop()
        self.player.setPosition(round(self.videoSlider.value()*self.player.duration()/self.maxValue))

    def slider_progress_released(self):
        self.player.play()
        self.timer.start()


    def display_page4(self):
        self.mVideoWin.hide()
        self.stackedWidget.setCurrentIndex(3)
    
    def track_live(self): 
        exVideo = self.myPath+"videos/init/record.avi"
        cameraCapture = cv2.VideoCapture(0)

        success, camera_frames = cameraCapture.read()
        fps = cameraCapture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print('============H * W',np.array(camera_frames).shape)  #(480, 640, 3)

        track_camera =  track_go(yolo=YOLO3(), target_video=exVideo)
        track_camera_out = cv2.VideoWriter(track_camera.out_dir + 'tracking_camera0' + '.avi', fourcc, fps, size)
        track_camera_out_REID = cv2.VideoWriter(track_camera.out_dir + 'tracking_camera0_REID' + '.avi', fourcc, fps, size)
    
        while self.liveButton.isChecked():
            if cv2.waitKey(1) == 27:
                break
            print('==========Tracking and REID Processing....')
            frame_result, frame_result_REID= track_camera.track_ID(camera_frames)
            # if args.save_result2video: 
            track_camera_out.write(frame_result)
            cv2.waitKey(1)
            Is_REID = True
            if Is_REID:
                track_camera_out_REID.write(frame_result_REID)
                cv2.waitKey(1)
            width = 640
            height = 480
            self.c4Label.clear()
            img = cv2.cvtColor(camera_frames, cv2.COLOR_BGR2RGB)
            label_data = QtGui.QImage(img.data, width, height, width*3, QtGui.QImage.Format_RGB888)  
            self.c4Label.setPixmap(QtGui.QPixmap.fromImage(label_data))
        
            success, camera_frames = cameraCapture.read()

        cameraCapture.release()
        self.c4Label.clear()

    def about(self):
        QMessageBox.about(
            self,
            "About Sample Editor",
            "<p>A sample text editor app built with:</p>"
            "<p>- PyQt</p>"
            "<p>- Qt Designer</p>"
            "<p>- Python</p>",
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
