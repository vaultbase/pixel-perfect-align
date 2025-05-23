#!/usr/bin/env python3
"""
Pixel Perfect Align - Simple GUI
Zero-effort image alignment tool
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
from typing import List, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap, QDragEnterEvent, QDropEvent

from src.core.pipeline import AlignmentPipeline
from src.utils.io import ImageLoader, ResultExporter
from src.utils.logging import setup_logging


class AlignmentThread(QThread):
    """Worker thread for alignment processing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[Path], output_dir: Path):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.loader = ImageLoader(max_resolution=None)
        self.pipeline = AlignmentPipeline(overlap_ratio=0.5)
    
    def run(self):
        try:
            # Load images
            self.progress.emit("Loading images...")
            images = []
            metadata = []
            
            for i, path in enumerate(self.image_paths):
                self.progress.emit(f"Loading {path.name} ({i+1}/{len(self.image_paths)})")
                try:
                    img, meta = self.loader.load_image(path)
                    images.append(img)
                    metadata.append(meta)
                except Exception as e:
                    self.progress.emit(f"⚠️ Skipping {path.name}: {str(e)}")
                    continue
            
            # Check if we have enough images
            if len(images) < 2:
                self.error.emit("Not enough valid images for alignment. Need at least 2 images.")
                return
            
            # Run alignment
            self.progress.emit(f"Running alignment pipeline with {len(images)} images...")
            self.progress.emit("Stage 1/5: Initial Fourier alignment...")
            results = self.pipeline.align(images, metadata)
            
            # Export results
            self.progress.emit("Exporting results...")
            exporter = ResultExporter(self.output_dir)
            
            # Export aligned images
            self.progress.emit("Saving aligned images...")
            aligned_paths = exporter.export_aligned_images(
                results['aligned_images'],
                results['canvas_size'],
                metadata
            )
            
            # Always export transforms
            self.progress.emit("Saving transformation data...")
            transform_path = exporter.export_transforms(results['transforms'])
            
            # Generate composite
            self.progress.emit("Generating composite image...")
            composite_path = exporter.generate_composite(
                results['aligned_images'],
                results['masks']
            )
            
            # Save summary
            summary = {
                'canvas_size': results['canvas_size'],
                'num_images': len(images),
                'metrics': results['metrics'],
                'aligned_images': [str(p) for p in aligned_paths],
                'composite': str(composite_path),
                'transforms': str(transform_path),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.finished.emit(summary)
            
        except Exception as e:
            self.error.emit(str(e))


class PixelPerfectAlignGUI(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.selected_folder = None
        self.selected_images = []
        self.alignment_thread = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Pixel Perfect Align")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set modern style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0051D5;
            }
            QPushButton:pressed {
                background-color: #003F9F;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 5px;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 10px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 6px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 6px;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Pixel Perfect Align")
        title_font = QFont()
        title_font.setPointSize(28)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Zero-effort image alignment for Fujifilm GFX100S II")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #666666; font-size: 16px;")
        main_layout.addWidget(subtitle_label)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Image selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Folder selection group
        folder_group = QGroupBox("Select Images")
        folder_layout = QVBoxLayout()
        
        # Browse button
        self.browse_btn = QPushButton("📁 Browse for Folder")
        self.browse_btn.clicked.connect(self.browse_folder)
        self.browse_btn.setMinimumHeight(50)
        folder_layout.addWidget(self.browse_btn)
        
        # Or label
        or_label = QLabel("— OR —")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        or_label.setStyleSheet("color: #999999; margin: 10px 0;")
        folder_layout.addWidget(or_label)
        
        # Select files button
        self.select_files_btn = QPushButton("📄 Select Individual Images")
        self.select_files_btn.clicked.connect(self.select_files)
        self.select_files_btn.setMinimumHeight(50)
        folder_layout.addWidget(self.select_files_btn)
        
        # Drop zone
        self.drop_label = QLabel("Or drag & drop images here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 10px;
                padding: 30px;
                background-color: #fafafa;
                color: #999999;
                font-size: 16px;
            }
        """)
        self.drop_label.setMinimumHeight(100)
        folder_layout.addWidget(self.drop_label)
        
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)
        
        # Image list
        list_group = QGroupBox("Selected Images")
        list_layout = QVBoxLayout()
        
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        list_layout.addWidget(self.image_list)
        
        # Image count label
        self.count_label = QLabel("No images selected")
        self.count_label.setStyleSheet("color: #666666; margin-top: 5px;")
        list_layout.addWidget(self.count_label)
        
        list_group.setLayout(list_layout)
        left_layout.addWidget(list_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Processing
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Status group
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout()
        
        # Big align button
        self.align_btn = QPushButton("🚀 Start Alignment")
        self.align_btn.clicked.connect(self.start_alignment)
        self.align_btn.setEnabled(False)
        self.align_btn.setMinimumHeight(60)
        self.align_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: #34C759;
            }
            QPushButton:hover {
                background-color: #2FB24C;
            }
            QPushButton:pressed {
                background-color: #2A9D3F;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        status_layout.addWidget(self.align_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_label = QLabel("Ready to align images")
        self.status_label.setStyleSheet("font-size: 16px; margin: 10px 0;")
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Results group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.results_label = QLabel("No results yet")
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)
        
        self.open_folder_btn = QPushButton("📂 Open Output Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)
        results_layout.addWidget(self.open_folder_btn)
        
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Add initial log message
        self.log("Pixel Perfect Align ready!")
        self.log("Select a folder or individual images to begin.")
    
    def browse_folder(self):
        """Browse for a folder containing images"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Images",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.load_folder(Path(folder))
    
    def select_files(self):
        """Select individual image files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            str(Path.home()),
            "Image Files (*.tif *.tiff *.jpg *.jpeg *.png *.raw *.dng *.raf *.nef *.cr2 *.arw)"
        )
        
        if files:
            self.load_files([Path(f) for f in files])
    
    def load_folder(self, folder: Path):
        """Load all images from a folder"""
        self.selected_folder = folder
        
        # Find all image files
        image_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.raw', '.dng', '.raf', '.nef', '.cr2', '.arw'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        # Filter out hidden/system files and duplicates
        image_files = [f for f in set(image_files) if not f.name.startswith('._') and not f.name.startswith('.')]
        
        # Sort by name
        image_files = sorted(image_files)
        
        if image_files:
            self.load_files(image_files)
            self.log(f"Loaded {len(image_files)} images from {folder.name}")
        else:
            QMessageBox.warning(self, "No Images", "No supported image files found in the selected folder.")
    
    def load_files(self, files: List[Path]):
        """Load individual image files"""
        self.selected_images = files
        
        # Update list widget
        self.image_list.clear()
        for file in files:
            item = QListWidgetItem(file.name)
            item.setData(Qt.ItemDataRole.UserRole, str(file))
            self.image_list.addItem(item)
        
        # Update count
        self.count_label.setText(f"{len(files)} images selected")
        
        # Enable align button
        self.align_btn.setEnabled(len(files) >= 2)
        
        if len(files) < 2:
            self.status_label.setText("Need at least 2 images to align")
        else:
            self.status_label.setText(f"Ready to align {len(files)} images")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events"""
        files = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.raw', '.dng', '.raf', '.nef', '.cr2', '.arw'}:
                # Skip hidden/system files
                if not path.name.startswith('._') and not path.name.startswith('.'):
                    files.append(path)
            elif path.is_dir():
                self.load_folder(path)
                return
        
        if files:
            self.load_files(files)
    
    def start_alignment(self):
        """Start the alignment process"""
        if not self.selected_images:
            return
        
        # Determine output directory
        if self.selected_folder:
            output_dir = self.selected_folder / "Aligned"
        else:
            # Use parent directory of first image
            output_dir = self.selected_images[0].parent / "Aligned"
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        self.output_dir = output_dir
        
        # Update UI
        self.align_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.select_files_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Clear log
        self.log_text.clear()
        self.log("Starting alignment process...")
        self.log(f"Output directory: {output_dir}")
        
        # Create and start worker thread
        self.alignment_thread = AlignmentThread(self.selected_images, output_dir)
        self.alignment_thread.progress.connect(self.on_progress)
        self.alignment_thread.finished.connect(self.on_finished)
        self.alignment_thread.error.connect(self.on_error)
        self.alignment_thread.start()
    
    def on_progress(self, message: str):
        """Handle progress updates"""
        self.status_label.setText(message)
        self.log(message)
    
    def on_finished(self, summary: dict):
        """Handle successful completion"""
        # Update UI
        self.progress_bar.setVisible(False)
        self.align_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.select_files_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        
        # Show results
        self.status_label.setText("✅ Alignment completed successfully!")
        
        metrics = summary['metrics']
        results_text = f"""
<b>Alignment Results:</b><br>
• Canvas size: {summary['canvas_size'][1]} × {summary['canvas_size'][0]} pixels<br>
• Images aligned: {summary['num_images']}<br>
• Average error: {metrics['avg_error']:.3f} pixels<br>
• Maximum error: {metrics['max_error']:.3f} pixels<br>
<br>
<b>Output files saved in:</b><br>
{self.output_dir}
        """
        self.results_label.setText(results_text)
        
        self.log("\n✅ ALIGNMENT COMPLETED SUCCESSFULLY!")
        self.log(f"Average alignment error: {metrics['avg_error']:.3f} pixels")
        self.log(f"All files saved to: {self.output_dir}")
        
        # Show success message
        QMessageBox.information(
            self,
            "Success",
            f"Alignment completed!\n\nImages have been saved to:\n{self.output_dir}\n\nAverage error: {metrics['avg_error']:.3f} pixels"
        )
    
    def on_error(self, error: str):
        """Handle errors"""
        # Update UI
        self.progress_bar.setVisible(False)
        self.align_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.select_files_btn.setEnabled(True)
        
        self.status_label.setText("❌ Alignment failed!")
        self.log(f"\n❌ ERROR: {error}")
        
        # Show error message
        QMessageBox.critical(
            self,
            "Alignment Error",
            f"An error occurred during alignment:\n\n{error}\n\nPlease check the log for details."
        )
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        if hasattr(self, 'output_dir') and self.output_dir.exists():
            if sys.platform == 'darwin':  # macOS
                os.system(f'open "{self.output_dir}"')
            elif sys.platform == 'win32':  # Windows
                os.startfile(str(self.output_dir))
            else:  # Linux
                os.system(f'xdg-open "{self.output_dir}"')
    
    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Pixel Perfect Align")
    
    # Set application style
    app.setStyle('Fusion')
    
    window = PixelPerfectAlignGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()