import sys
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QLineEdit, QRadioButton, QPushButton

class GUI(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HAHA (HIHI)")
        self.setFixedSize(400, 400)

        # Gridding definitions
        label_gridtype = QLabel("Please choose type of grid:")
        self.gridtype = QRadioButton("Square")
        self.gridtype.setChecked(True)
        self.gridtype2 = QRadioButton("Hex")
        self.gridtype3 = QRadioButton("Tri")
        label_dimensionality = QLabel("Please choose domain dimensionality:")
        self.dimensionality = QRadioButton("2")
        self.dimensionality.setChecked(True)
        self.dimensionality2 = QRadioButton("3")
        label_xmin = QLabel("X-coordinate of the start of simulation domain:")
        self.xmin = QLineEdit()
        label_xmax = QLabel("X-coordinate of the end of simulation domain:")
        self.xmax = QLineEdit()
        label_xinc = QLabel("X-coordinate increments in the simulation domain:")
        self.xinc = QLineEdit()
        label_ymin = QLabel("Y-coordinate of the start of simulation domain:")
        self.ymin = QLineEdit()
        label_ymax = QLabel("Y-coordinate of the end of simulation domain:")
        self.ymax = QLineEdit()
        label_yinc = QLabel("Y-coordinate increments in the simulation domain:")
        self.yinc = QLineEdit()
        label_zmin = QLabel("Z-coordinate of the start of simulation domain:")
        self.zmin = QLineEdit()
        label_zmax = QLabel("Z-coordinate of the end of simulation domain:")
        self.zmax = QLineEdit()
        label_zinc = QLabel("Z-coordinate increments in the simulation domain:")
        self.zinc = QLineEdit()
        label_transformation = QLabel("Geometric transformation operation for the grid:")
        self.transformation = QLineEdit()

        # Listeners
        self.gridtype.clicked.connect(self.on_gridtype_clicked)
        self.dimensionality.clicked.connect(self.on_dimensionality_clicked)

    def on_gridtype_clicked(self):
        if self.gridtype.isChecked():
            self.gridtype2.setChecked(False)
            self.gridtype3.setChecked(False)
        elif self.gridtype2.isChecked():
            self.gridtype.setChecked(False)
            self.gridtype3.setChecked(False)
        elif self.gridtype3.isChecked():
            self.gridtype.setChecked(False)
            self.gridtype2.setChecked(False)

    def on_dimensionality_clicked(self):
        if self.dimensionality.isChecked():
            self.dimensionality2.setChecked(False)
        elif self.dimensionality2.isChecked():
            self.dimensionality.setChecked(False)

    def show(self):
        super().show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec())
