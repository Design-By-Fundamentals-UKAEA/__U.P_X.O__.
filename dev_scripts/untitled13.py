import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QPushButton


class DataEntryPage(QWidget):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title label
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 14px; font-family: Arial;")
        layout.addWidget(title_label)

        # Data entry widgets
        self.widgets = []

        for target in self.get_target_variables():
            label = QLabel(f"{target['index']}. {target['label']}")
            label.setStyleSheet("font-size: 14px; font-family: Arial;")
            layout.addWidget(label)

            widget = QLineEdit() if target['input_type'] == 'numeric' else QWidget()

            if target['input_type'] == 'radio':
                radio_layout = QHBoxLayout()
                for option in target['options']:
                    radio_btn = QRadioButton(option)
                    radio_layout.addWidget(radio_btn)
                    if option == target['default']:
                        radio_btn.setChecked(True)
                widget.setLayout(radio_layout)

            layout.addWidget(widget)
            self.widgets.append(widget)

        self.setLayout(layout)

    def get_target_variables(self):
        targets_page1 = [
            {"index": 1, "label": "Please choose type of grid.", "input_type": "radio", "options": ["square", "hex", "tri"], "default": "square"},
            {"index": 2, "label": "Please choose domain dimensionality", "input_type": "radio", "options": ["2", "3"], "default": "2"},
            {"index": 3, "label": "X-coordinate of the start of simulation domain", "input_type": "numeric"},
            {"index": 4, "label": "X-coordinate of the end of simulation domain", "input_type": "numeric"},
            {"index": 5, "label": "X-coordinate increments in the simulation domain", "input_type": "numeric"},
            {"index": 6, "label": "Y-coordinate of the start of simulation domain", "input_type": "numeric"},
            {"index": 7, "label": "Y-coordinate of the end of simulation domain", "input_type": "numeric"},
            {"index": 8, "label": "Y-coordinate increments in the simulation domain", "input_type": "numeric"},
            {"index": 9, "label": "Z-coordinate of the start of simulation domain", "input_type": "numeric"},
            {"index": 10, "label": "Z-coordinate of the end of simulation domain", "input_type": "numeric"},
            {"index": 11, "label": "Z-coordinate increments in the simulation domain", "input_type": "numeric"},
            {"index": 12, "label": "Geometric transformation operation for the grid", "input_type": "numeric"},
        ]

        targets_page2 = [
            {"index": 1, "label": "Number of individual state values", "input_type": "numeric"},
            {"index": 2, "label": "Number of Monte-Carlo iterations", "input_type": "numeric"},
        ]

        if "PAGE 1" in self.title:
            return targets_page1
        else:
            return targets_page2


class DataEntryForm(QWidget):
    def __init__(self):
        super().__init__()

        self.pages = [DataEntryPage("HAHA (HIHI) - PAGE 1"), DataEntryPage("HAHA (HIHI) - PAGE 2")]
        self.current_page = 0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.stack = QWidget()
        self.stack_layout = QVBoxLayout()
        self.stack.setLayout(self.stack_layout)

        for page in self.pages:
            self.stack_layout.addWidget(page)
            page.setVisible(False)

        self.pages[0].setVisible(True)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_page)

        layout.addWidget(self.stack)
        layout.addWidget(self.next_button, alignment=0x0002)

        self.setLayout(layout)
        self.setWindowTitle("HAHA (HIHI)")
        self.setFixedSize(400, 400)

    def next_page(self):
        current_widget = self.pages[self.current_page]
        next_widget = self.pages[self.current_page + 1]

        for widget in current_widget.widgets:
            if isinstance(widget, QLineEdit):
                # Add data to data class here if needed
                pass

        self.stack_layout.setCurrentWidget(next_widget)
        self.current_page += 1

        if self.current_page == len(self.pages) - 1:
            self.next_button.setText("Save Data")

        if self.current_page == len(self.pages):
            # Save data or process the collected information
            # For this example, we will just print the data
            print("Data saved:")
            for page in self.pages:
                for widget in page.widgets:
                    if isinstance(widget, QLineEdit):
                        print(widget.text())
            QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataEntryForm()
    window.show()
    sys.exit(app.exec_())
