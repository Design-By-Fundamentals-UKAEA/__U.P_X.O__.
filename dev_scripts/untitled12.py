# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 03:29:44 2023

@author: rg5749
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton
from dataclasses import dataclass

@dataclass(init=True, repr=True, eq=True)
class mcgs_gridding_definitions:
    gridtype: str
    dimensionality: int
    xmin: float
    xmax: float
    xinc: float
    ymin: float
    ymax: float
    yinc: float
    zmin: float
    zmax: float
    zinc: float
    transformation: str

class DataEntryForm(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.gridtype_input = QLineEdit()
        self.dimensionality_input = QLineEdit()
        self.xmin_input = QLineEdit()
        self.xmax_input = QLineEdit()
        self.xinc_input = QLineEdit()
        self.ymin_input = QLineEdit()
        self.ymax_input = QLineEdit()
        self.yinc_input = QLineEdit()
        self.zmin_input = QLineEdit()
        self.zmax_input = QLineEdit()
        self.zinc_input = QLineEdit()
        self.transformation_input = QLineEdit()

        layout.addWidget(QLabel("Grid Type:"))
        layout.addWidget(self.gridtype_input)

        layout.addWidget(QLabel("Dimensionality:"))
        layout.addWidget(self.dimensionality_input)

        layout.addWidget(QLabel("X-min:"))
        layout.addWidget(self.xmin_input)

        layout.addWidget(QLabel("X-max:"))
        layout.addWidget(self.xmax_input)

        layout.addWidget(QLabel("X-increment:"))
        layout.addWidget(self.xinc_input)

        layout.addWidget(QLabel("Y-min:"))
        layout.addWidget(self.ymin_input)

        layout.addWidget(QLabel("Y-max:"))
        layout.addWidget(self.ymax_input)

        layout.addWidget(QLabel("Y-increment:"))
        layout.addWidget(self.yinc_input)

        layout.addWidget(QLabel("Z-min:"))
        layout.addWidget(self.zmin_input)

        layout.addWidget(QLabel("Z-max:"))
        layout.addWidget(self.zmax_input)

        layout.addWidget(QLabel("Z-increment:"))
        layout.addWidget(self.zinc_input)

        layout.addWidget(QLabel("Transformation:"))
        layout.addWidget(self.transformation_input)

        save_button = QPushButton("Save Data")
        save_button.clicked.connect(self.save_data)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def save_data(self):
        data = mcgs_gridding_definitions(
            gridtype=self.gridtype_input.text(),
            dimensionality=int(self.dimensionality_input.text()),
            xmin=float(self.xmin_input.text()),
            xmax=float(self.xmax_input.text()),
            xinc=float(self.xinc_input.text()),
            ymin=float(self.ymin_input.text()),
            ymax=float(self.ymax_input.text()),
            yinc=float(self.yinc_input.text()),
            zmin=float(self.zmin_input.text()),
            zmax=float(self.zmax_input.text()),
            zinc=float(self.zinc_input.text()),
            transformation=self.transformation_input.text()
        )

        # Do something with the data, e.g., print it
        print(data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataEntryForm()
    window.show()
    sys.exit(app.exec_())
