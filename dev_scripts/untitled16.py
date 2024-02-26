import sys
from dataclasses import dataclass
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox
from PyQt5.QtCore import Qt  # Import the QtCore module


@dataclass(init=True, repr=True, eq=True)
class mcgs_gridding_definitions:
    gridtype: str  # Type of underlying grid
    dimensionality: int  # Physical dimensionality of the domain
    xmin: float  # X-coordinate of the start of the simulation domain
    xmax: float  # X-coordinate of the end of the simulation domain
    xinc: float  # X-coordinate increments in the simulation domain
    ymin: float  # Y-coordinate of the start of the simulation domain
    ymax: float  # Y-coordinate of the end of the simulation domain
    yinc: float  # Y-coordinate increments in the simulation domain
    zmin: float  # Z-coordinate of the start of the simulation domain
    zmax: float  # Z-coordinate of the end of the simulation domain
    zinc: float  # Z-coordinate increments in the simulation domain
    transformation: str  # Geometric transformation operation for the grid


class DataEntryForm(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.gridtype_label = QLabel("Please choose type of grid:")
        self.gridtype_combo = QComboBox()
        self.gridtype_combo.addItems(["Square", "Hex", "Tri"])
        self.gridtype_combo.setCurrentIndex(0)  # Set default value to Square

        gridtype_layout = QHBoxLayout()
        gridtype_layout.addWidget(self.gridtype_combo)

        layout.addWidget(self.gridtype_label)
        layout.addLayout(gridtype_layout)

        self.dimensionality_label = QLabel("Please choose domain dimensionality:")
        self.dimensionality_combo = QComboBox()
        self.dimensionality_combo.addItems(["2", "3"])
        self.dimensionality_combo.setCurrentIndex(0)  # Set default value to 2

        dimensionality_layout = QHBoxLayout()
        dimensionality_layout.addWidget(self.dimensionality_combo)

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

        default_values = [
            ("xmin", -1),
            ("xmax", 1),
            ("xinc", 0.2),
            ("ymin", -1),
            ("ymax", 1),
            ("yinc", 0.2),
            ("zmin", -1),
            ("zmax", 1),
            ("zinc", 0.2),
            ("transformation", ""),
        ]

        for i, label in enumerate(variable_labels):
            input_label = QLabel(label)
            input_field = QLineEdit(str(default_values[i][1]))
            self.variable_entries[default_values[i][0]] = input_field

            layout.addWidget(input_label)
            layout.addWidget(input_field)

        save_exit_button = QPushButton("Save and Exit")
        save_exit_button.clicked.connect(self.save_and_exit)

        layout.addWidget(save_exit_button)

        self.setLayout(layout)
        self.setWindowTitle("HAHA (HIHI)")
        self.setFixedSize(400, 400)
        self.show()

    def save_and_exit(self):
        data = mcgs_gridding_definitions(
            gridtype=self.gridtype_combo.currentText(),
            dimensionality=int(self.dimensionality_combo.currentText()),
            xmin=float(self.variable_entries["xmin"].text()),
            xmax=float(self.variable_entries["xmax"].text()),
            xinc=float(self.variable_entries["xinc"].text()),
            ymin=float(self.variable_entries["ymin"].text()),
            ymax=float(self.variable_entries["ymax"].text()),
            yinc=float(self.variable_entries["yinc"].text()),
            zmin=float(self.variable_entries["zmin"].text()),
            zmax=float(self.variable_entries["zmax"].text()),
            zinc=float(self.variable_entries["zinc"].text()),
            transformation=self.variable_entries["transformation"].text(),
        )

        # Here, you can do whatever you want with the 'data' object, like saving it to a file or using it in your application.

        # Close the window upon clicking "Save and Exit"
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataEntryForm()
    sys.exit(app.exec_())
