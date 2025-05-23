#!/usr/bin/env python3
"""
Simple Tkinter GUI for Pixel Perfect Align
More stable alternative to PyQt6 on macOS
"""

import sys
import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

from src.core.pipeline import AlignmentPipeline
from src.utils.io import ImageLoader, ResultExporter


class PixelAlignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Perfect Align")
        self.root.geometry("800x600")
        
        # Variables
        self.input_folder = None
        self.processing = False
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_frame = ttk.Frame(self.root, padding="20")
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            title_frame,
            text="Pixel Perfect Align",
            font=('Arial', 24, 'bold')
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Simple image alignment for Fujifilm GFX100S II",
            font=('Arial', 12)
        )
        subtitle_label.pack()
        
        # Folder selection
        folder_frame = ttk.Frame(self.root, padding="20")
        folder_frame.pack(fill=tk.X)
        
        self.folder_label = ttk.Label(
            folder_frame,
            text="No folder selected",
            font=('Arial', 12)
        )
        self.folder_label.pack(pady=10)
        
        browse_btn = ttk.Button(
            folder_frame,
            text="Browse for Folder",
            command=self.browse_folder,
            style='Large.TButton'
        )
        browse_btn.pack(pady=10)
        
        # Align button
        self.align_btn = ttk.Button(
            folder_frame,
            text="Start Alignment",
            command=self.start_alignment,
            state=tk.DISABLED,
            style='Success.TButton'
        )
        self.align_btn.pack(pady=10)
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(
            self.root,
            textvariable=self.progress_var,
            font=('Arial', 12)
        )
        progress_label.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(pady=10)
        
        # Log area
        log_frame = ttk.Frame(self.root, padding="20")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        log_label = ttk.Label(log_frame, text="Log:", font=('Arial', 12))
        log_label.pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            width=80,
            font=('Courier', 10),
            bg='black',
            fg='lime'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Style configuration
        style = ttk.Style()
        style.configure('Large.TButton', font=('Arial', 14))
        style.configure('Success.TButton', font=('Arial', 16, 'bold'))
        
    def browse_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder with images",
            initialdir=Path.home()
        )
        
        if folder:
            self.input_folder = Path(folder)
            self.folder_label.config(text=f"Selected: {self.input_folder.name}")
            self.align_btn.config(state=tk.NORMAL)
            self.log(f"Selected folder: {self.input_folder}")
            
            # Count images
            images = self.find_images()
            self.log(f"Found {len(images)} images")
            
    def find_images(self):
        if not self.input_folder:
            return []
            
        valid_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
        images = []
        
        for ext in valid_extensions:
            images.extend(self.input_folder.glob(f"*{ext}"))
            images.extend(self.input_folder.glob(f"*{ext.upper()}"))
        
        # Filter out hidden files
        images = [img for img in images if not img.name.startswith('.') and not img.name.startswith('._')]
        
        return sorted(set(images))
    
    def start_alignment(self):
        if self.processing:
            return
            
        self.processing = True
        self.align_btn.config(state=tk.DISABLED)
        self.progress_bar.start(10)
        
        # Run in thread to keep GUI responsive
        thread = threading.Thread(target=self.run_alignment)
        thread.daemon = True
        thread.start()
        
    def run_alignment(self):
        try:
            self.update_progress("Finding images...")
            images = self.find_images()
            
            if len(images) < 2:
                self.show_error("Need at least 2 images to align")
                return
                
            # Create output directory
            output_dir = self.input_folder / "Aligned"
            output_dir.mkdir(exist_ok=True)
            self.log(f"Output directory: {output_dir}")
            
            # Load images
            self.update_progress("Loading images...")
            loader = ImageLoader()
            loaded_images = []
            metadata = []
            
            for i, img_path in enumerate(images):
                self.update_progress(f"Loading {img_path.name} ({i+1}/{len(images)})")
                try:
                    img, meta = loader.load_image(img_path)
                    loaded_images.append(img)
                    metadata.append(meta)
                except Exception as e:
                    self.log(f"⚠️ Skipping {img_path.name}: {e}")
            
            if len(loaded_images) < 2:
                self.show_error("Not enough valid images after loading")
                return
                
            # Run alignment
            self.update_progress("Running alignment...")
            pipeline = AlignmentPipeline(overlap_ratio=0.5)
            results = pipeline.align(loaded_images, metadata)
            
            # Export results
            self.update_progress("Saving results...")
            exporter = ResultExporter(output_dir)
            
            exporter.export_aligned_images(
                results['aligned_images'],
                results['canvas_size'],
                metadata
            )
            
            exporter.export_transforms(results['transforms'])
            
            exporter.generate_composite(
                results['aligned_images'],
                results['masks']
            )
            
            # Success
            self.update_progress("✅ Alignment completed!")
            self.log(f"Average error: {results['metrics']['avg_error']:.3f} pixels")
            self.log(f"Results saved to: {output_dir}")
            
            # Ask to open folder
            if messagebox.askyesno("Success", "Alignment completed!\n\nOpen output folder?"):
                if sys.platform == 'darwin':
                    os.system(f'open "{output_dir}"')
                elif sys.platform == 'win32':
                    os.startfile(str(output_dir))
                    
        except Exception as e:
            self.show_error(f"Error during alignment: {e}")
            import traceback
            self.log(traceback.format_exc())
            
        finally:
            self.processing = False
            self.progress_bar.stop()
            self.align_btn.config(state=tk.NORMAL)
            
    def update_progress(self, message):
        self.progress_var.set(message)
        self.log(message)
        self.root.update()
        
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def show_error(self, message):
        self.log(f"❌ {message}")
        messagebox.showerror("Error", message)


def main():
    root = tk.Tk()
    app = PixelAlignGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()