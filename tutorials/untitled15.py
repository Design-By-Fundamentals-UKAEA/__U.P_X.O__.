import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QPushButton
from PyQt5.QtCore import Qt  # Import the QtCore module
from dataclasses import dataclass

@dataclass(init=True, repr=True, eq=True)
class mcgs_gridding_definitions:
    gridtype: str  # Type of underlying grid
    dimensionality: int  # Physical dimensionality of the domain
    xmin: float  # X-coordinate of the start of simulation domain
    xmax: float  # X-coordinate of the end of simulation domain
    xinc: float  # X-coordinate increments in the simulation domain
    ymin: float  # Y-coordinate of the start of simulation domain
    ymax: float  # Y-coordinate of the end of simulation domain
    yinc: float  # Y-coordinate increments in the simulation domain
    zmin: float  # Z-coordinate of the start of simulation domain
    zmax: float  # Z-coordinate of the end of simulation domain
    zinc: float  # Z-coordinate increments in the simulation domain
    transformation: str  # Geometric transformation operation for the grid

class DataEntryForm(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.gridtype_label = QLabel("Please choose type of grid:")
        self.gridtype_radio_square = QRadioButton("Square")
        self.gridtype_radio_hex = QRadioButton("Hex")
        self.gridtype_radio_tri = QRadioButton("Tri")
        self.gridtype_radio_square.setChecked(True)  # Set default value to Square

        gridtype_layout = QHBoxLayout()
        gridtype_layout.addWidget(self.gridtype_radio_square)
        gridtype_layout.addWidget(self.gridtype_radio_hex)
        gridtype_layout.addWidget(self.gridtype_radio_tri)

        layout.addWidget(self.gridtype_label)
        layout.addLayout(gridtype_layout)

        self.dimensionality_label = QLabel("Please choose domain dimensionality:")
        self.dimensionality_radio_2 = QRadioButton("2")
        self.dimensionality_radio_3 = QRadioButton("3")
        self.dimensionality_radio_2.setChecked(True)  # Set default value to 2

        dimensionality_layout = QHBoxLayout()
        dimensionality_layout.addWidget(self.dimensionality_radio_2)
        dimensionality_layout.addWidget(self.dimensionality_radio_3)

        layout.addWidget(self.dimensionality_label)
        layout.addLayout(dimensionality_layout)

        self.variable_entries = {}

        # Create input fields for the rest of the variables
        variable_labels = [
            "X-coordinate of the start of simulation domain",
            "X-coordinate of the end of simulation domain",
            "X-coordinate increments in the simulation domain",
            "Y-coordinate of the start of simulation domain",
            "Y-coordinate of the end of simulation domain",
            "Y-coordinate increments in the simulation domain",
            "Z-coordinate of the start of simulation domain",
            "Z-coordinate of the end of simulation domain",
            "Z-coordinate increments in the simulation domain",
            "Geometric transformation operation for the grid",
        ]

        for i, label in enumerate(variable_labels):
            input_label = QLabel(f"{i+1}. {label}")
            input_field = QLineEdit()
            self.variable_entries[i] = input_field
            layout.addWidget(input_label)
            layout.addWidget(input_field)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.save_data)

        layout.addWidget(self.next_button, alignment=Qt.AlignRight)  # Use QtCore.Qt.AlignRight for alignment

        self.setLayout(layout)

        self.setWindowTitle("HAHA (HIHI)")
        self.setFixedSize(400, 600)  # Set a constant window size

    def save_data(self):
        # Retrieve entered data from the input fields
        gridtype = "Square" if self.gridtype_radio_square.isChecked() else "Hex" if self.gridtype_radio_hex.isChecked() else "Tri"
        dimensionality = int(self.dimensionality_radio_2.isChecked()) if 2 else int(self.dimensionality_radio_3.isChecked()) if 3 else 2
        xmin = float(self.variable_entries[0].text())
        xmax = float(self.variable_entries[1].text())
        xinc = float(self.variable_entries[2].text())
        ymin = float(self.variable_entries[3].text())
        ymax = float(self.variable_entries[4].text())
        yinc = float(self.variable_entries[5].text())
        zmin = float(self.variable_entries[6].text())
        zmax = float(self.variable_entries[7].text())
        zinc = float(self.variable_entries[8].text())
        transformation = self.variable_entries[9].text()

        # Create the data class object here using the entered data
        mcgs_gridding_def = mcgs_gridding_definitions(
            gridtype=gridtype,
            dimensionality=dimensionality,
            xmin=xmin,
            xmax=xmax,
            xinc=xinc,
            ymin=ymin,
            ymax=ymax,
            yinc=yinc,
            zmin=zmin,
            zmax=zmax,
            zinc=zinc,
            transformation=transformation
        )

        # Display or process the data as needed
        print(mcgs_gridding_def)

        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataEntryForm()
    window.show()
    sys.exit(app.exec_())
