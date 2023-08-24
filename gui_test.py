#pyuic5 -o D:/CC/APP/Person_Tracking/GUI/gui.py D:/CC/APP/Person_Tracking/GUI/gui.ui
#  D:/CC/APP/GUI/gui.ui

import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from GUI.gui import Ui_MainWindow


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self) # 初始化组件和布局
    # def connectSignalsSlots(self):
    #     self.actionAbout.triggered.connect(self.about) # 点击 about 会触发介绍
    #     self.actionFind.triggered.connect(self.findAndReplace) # 点击 find 触发一个 Dialog
    # def findAndReplace(self):
    #     dialog = FindReplaceDialog(self)
    #     dialog.show()
    #     dialog.exec()
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

    

