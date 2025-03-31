import sys
import mss
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QLineEdit, QGroupBox, QHBoxLayout, QFrame, QMessageBox, QStyledItemDelegate, QSizePolicy, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QStandardItem, QStandardItemModel
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from pynput import mouse
import time
import json

from utils import load_model

dropdown_categories = [
    ("▶️ Classification Models", [
        {'name': "Tumor Compact (VGG19)", 'info_file': 'metadata/tumor_compact_vgg.json'},
        {'name': "Tumor Compact (EfficientNetV2) (Test)", 'info_file': 'metadata/tumor_compact_efficientnet.json'}
    ]),
    ("▶️ Segmentation Models", [
        {'name': "MIB (YOLO)", 'info_file': 'metadata/mib_yolo.json'}
    ]),
]



########################################################################

# Mapping of model name to metadata data
model_to_info = {}
for _, _v in dropdown_categories:
    for __v in _v:
        with open(__v['info_file']) as f:
            model_to_info[__v['name']] = json.load(f)



# Worker thread for continuous image classification
class ClassificationThread(QThread):
    update_image = pyqtSignal(np.ndarray, str)

    def __init__(self, ui_instance, model_name):
        super().__init__()
        self.ui_instance = ui_instance # So that the selected_region field can be updated and used here in real-time
        self.model_name = model_name
        self.running = True

        self.model, self.process_region = load_model(model_to_info[model_name])
        self.metadata = model_to_info[model_name]

    def run(self):
        while self.running:
            start = time.time()
            frame, result = self.process_region(self.ui_instance.selected_region, **{'model': self.model, 'metadata': self.metadata})
            result += '\n({:.2f} sec)'.format(time.time() - start)

            self.update_image.emit(frame, result)
            time.sleep(0.1)  # Adjust refresh rate as needed

    def stop(self):
        self.running = False


# GUI Application
class ImageClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_region = None
        self.thread = None


    def initUI(self):
        """Initialize GUI components"""
        self.setWindowTitle("Realtime Inference")
        # self.setGeometry(100, 100, 600, 400)

        # main_layout_main = QHBoxLayout()
        # main_layout_main.addSpacing(10)

        main_layout_l = QVBoxLayout()
        # main_layout.addSpacing(10)

        main_layout_r = QVBoxLayout()





        ##############################################################################

        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()

        ### Dropdown

        class PaddedItemDelegate(QStyledItemDelegate):
            """Custom delegate to add padding/margin to QComboBox items."""

            PADDING = 10  # Adjust padding value for more/less space

            def paint(self, painter, option, index):
                """Modify the item rectangle to create padding effect."""
                option.rect = QRect(
                    option.rect.x() + self.PADDING,  # Left padding
                    option.rect.y() + self.PADDING // 2,  # Top padding
                    option.rect.width() - 2 * self.PADDING,  # Right padding
                    option.rect.height() - self.PADDING  # Bottom padding
                )
                super().paint(painter, option, index)

            def sizeHint(self, option, index):
                """Increase item height to prevent squishing."""
                size = super().sizeHint(option, index)
                return QSize(size.width(), size.height() + self.PADDING)

        # Dropdown for selecting a model
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.setItemDelegate(PaddedItemDelegate(self.model_dropdown))
        model_dropdown_items = QStandardItemModel()

        for category, models in dropdown_categories:
            # Create a bold header (non-selectable)
            header = QStandardItem(category)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setEnabled(False)  # Make it non-selectable
            model_dropdown_items.appendRow(header)

            # Add models under the header
            for m in models:
                model_dropdown_items.appendRow(QStandardItem(m['name']))

        self.model_dropdown.setModel(model_dropdown_items)
        self.model_dropdown.currentIndexChanged.connect(self.show_selected_model_info)
        model_layout.addWidget(self.model_dropdown)

        ### Class & Info button

        self.classes_button = QPushButton("Classes")
        self.classes_button.setFixedSize(100, 30)
        self.classes_button.clicked.connect(self.show_classes)  # Connect to classes popup function
        self.classes_button.setStyleSheet("QPushButton { margin-top: -5px;  }")  # Need-to to align with dropdown...
        model_layout.addWidget(self.classes_button)


        self.info_button = QPushButton("ℹ️️")
        self.info_button.setFixedSize(30, 30)  # Make it small
        self.info_button.clicked.connect(self.show_model_info)  # Connect to info popup function
        self.info_button.setStyleSheet("QPushButton { margin-top: -5px;  }") # Need-to to align with dropdown...
        model_layout.addWidget(self.info_button)



        model_group.setLayout(model_layout)
        main_layout_l.addWidget(model_group)


        ##############################################################################

        capture_group = QGroupBox("Screen Capture")
        capture_layout = QVBoxLayout()


        self.capture_recommendation = QLabel("Recommended Capture Size:")
        self.capture_recommendation.setStyleSheet("QLabel { color: red; }")
        capture_layout.addWidget(self.capture_recommendation)


        # Button to select screen region
        self.select_region_btn = QPushButton("Select Screen Region", self)
        self.select_region_btn.clicked.connect(self.select_screen_region)
        capture_layout.addWidget(self.select_region_btn)

        # Label to show selected screen region coordinates
        self.region_label = QLabel("Selected Region: NOT SET", self)
        self.region_label.setStyleSheet("QLabel { color: grey; }")
        capture_layout.addWidget(self.region_label)



        capture_group.setLayout(capture_layout)
        main_layout_l.addWidget(capture_group)

        # separator = QFrame()
        # separator.setFrameShape(QFrame.Shape.HLine)
        # separator.setFrameShadow(QFrame.Shadow.Sunken)
        # main_layout_l.addWidget(separator)

        self.show_selected_model_info()  # Force initial run (will be run everytime a model is selected from the dropdown)

        ##############################################################################

        classify_group = QGroupBox("Inference Controls")
        classify_layout = QVBoxLayout()

        # Button to start classification
        self.start_btn = QPushButton("Start", self)
        self.start_btn.clicked.connect(self.start_classification)
        classify_layout.addWidget(self.start_btn)
        self.start_btn.setEnabled(False)

        # Button to stop classification
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop_classification)
        classify_layout.addWidget(self.stop_btn)
        self.stop_btn.setEnabled(False)

        classify_group.setLayout(classify_layout)
        main_layout_r.addWidget(classify_group)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout_r.addWidget(separator)

        ##############################################################################

        display_group = QGroupBox("Inference Output")
        display_layout = QVBoxLayout()


        # Image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.image_label)

        # Classification result display
        self.result_label = QLabel("Result:")
        display_layout.addWidget(self.result_label)

        display_group.setLayout(display_layout)
        main_layout_r.addWidget(display_group)


        ##############################################################################

        model_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold;  }")
        capture_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold;  }")
        classify_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold;  }")
        display_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold;  }")


        # This is to make layout_l and layout_r side-by-side
        # Wrap layouts inside QWidget containers
        widget1 = QWidget()
        widget1.setLayout(main_layout_l)
        widget2 = QWidget()
        widget2.setLayout(main_layout_r)
        widget1.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        widget2.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Create the main HBoxLayout and add the widgets
        main_layout = QHBoxLayout()
        main_layout.addSpacing(10)
        main_layout.addWidget(widget1)
        main_layout.addWidget(widget2)


        self.setLayout(main_layout)

        # Automatically resize window to fit contents
        self.adjustSize()

        self.setFocus() # Prevents any textboxes from getting auto-focused when the application first starts

    def select_screen_region(self):


        def get_scaling_factor():
            import platform
            system = platform.system()

            if system == "Darwin":
                from AppKit import NSScreen, NSApplication

                NSApplication.sharedApplication().setActivationPolicy_(1)  # Prevents Dock icon bouncing

                # Gets the correct macOS Retina scaling factor using Pyobjc
                screen = NSScreen.mainScreen()
                return screen.backingScaleFactor()

            # Assumes Windows/Linux use 1.0 scaling by default (as mss typically works fine there)
            return 1.0

        def get_mouse_position():
            """Get the current mouse position using pynput."""
            pos = []

            def on_click(x, y, button, pressed):
                if pressed:
                    pos.append((x, y))
                    return False  # Stop listener

            with mouse.Listener(on_click=on_click) as listener:
                listener.join()

            return pos[0] if pos else (0, 0)

        x1, y1 = get_mouse_position()
        x2, y2 = get_mouse_position()

        self.selected_region = {
            "left": min(x1, x2),
            "top": min(y1, y2),
            "width": abs(x1-x2),
            "height": abs(y1-y2)
        }

        print(f"Selected region: {self.selected_region}")


        # Update the label with selected coordinates
        # self.region_label.setText(f"Selected Region: ({self.selected_region['left']:.2f}, {self.selected_region['top']:.2f}).\n"
        self.region_label.setText(f"Selected Region:\n"
                                  f"Width: {self.selected_region['width'] * get_scaling_factor():.2f} (px)\n"
                                  f"Height: {self.selected_region['height'] * get_scaling_factor():.2f} (px)")


        self.region_label.setStyleSheet("QLabel { color: green; }")
        self.start_btn.setEnabled(True)

    def start_classification(self):
        """Start the classification loop"""
        if self.selected_region is None:
            self.result_label.setText("Please select a region first!")
            return


        model_name = self.model_dropdown.currentText()
        self.thread = ClassificationThread(self, model_name)
        self.thread.update_image.connect(self.update_display)
        self.thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_classification(self):
        """Stop the classification thread"""
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_display(self, frame, result):
        """Update the GUI with the latest image and classification result"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        self.result_label.setText(f"Result: {result}")

    def closeEvent(self, event):
        """Ensure the thread stops when closing the app"""
        self.stop_classification()
        event.accept()

    def show_model_info(self):
        """Displays a popup with information about the selected model."""
        selected_model = self.model_dropdown.currentText()

        # Get the info text (default message if model not found)
        info_text = model_to_info[selected_model]['info']

        # Create and show the popup
        msg = QMessageBox(self)
        msg.setWindowTitle("Model Information")
        msg.setText(f"Information about {selected_model}:\n\n{info_text}")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()  # Show the popup

    def show_classes(self):
        """Displays a popup with information about the selected model's classes."""

        selected_model = self.model_dropdown.currentText()

        classes = sorted(model_to_info[selected_model]['classes'])

        from custom_widgets.TablePopup import TablePopup
        popup = TablePopup(self, items=classes, title="Classes")
        popup.exec()



    def show_selected_model_info(self):
        """Updates UI text based on the selected model."""
        selected_model = self.model_dropdown.currentText()

        tile_size = model_to_info[selected_model]['tile_size']

        res = f'This model was trained on images of size {tile_size} x {tile_size} (px) with 20X magnification.\n'
        res += 'Choosing a screen capture size smaller than these dimensions may result in inaccurate results.\n'
        res += 'Larger screen capture sizes will be sliced into smaller images and predictions aggregated.'

        # Update label text dynamically
        self.capture_recommendation.setText(f"Recommended Capture Size:\n{res}")



# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec())
