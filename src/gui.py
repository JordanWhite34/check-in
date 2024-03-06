import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict import predict_single_image, initialize_model, load_checkpoint  # Ensure these are correctly imported
from train import get_transforms
import torch

# Initialize the PyTorch model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = initialize_model().to(device)
checkpoint_path = 'checkpoints/best_checkpoint.pth.tar'
load_checkpoint(checkpoint_path, model, device)
transform = get_transforms()  # Adjust as necessary

class ImageClassifierApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Set up the button to load images
        self.loadImageButton = tk.Button(window, text="Load Image", command=self.load_image)
        self.loadImageButton.pack()

        # Label to display the prediction
        self.predictionLabel = tk.Label(window, text="Prediction: None", font=("Helvetica", 16))
        self.predictionLabel.pack()

        # Label to display the confidence
        self.confidenceLabel = tk.Label(window, text="Confidence: None", font=("Helvetica", 16))
        self.confidenceLabel.pack()

        # The image display label
        self.imageDisplayLabel = tk.Label(window)
        self.imageDisplayLabel.pack()

        self.window.mainloop()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:  # If the user selects a file
            img = Image.open(file_path)
            img.thumbnail((250, 250))  # Resize for display
            photo = ImageTk.PhotoImage(img)
            self.imageDisplayLabel.configure(image=photo)
            self.imageDisplayLabel.image = photo  # Keep a reference!

            class_label, confidence = predict_single_image(file_path, model, device, transform)
            self.predictionLabel.configure(text=f"Prediction: {class_label}")
            self.confidenceLabel.configure(text=f"Confidence: {100*confidence:.2f}%")


if __name__ == "__main__":
    # Create the application window
    root = tk.Tk()
    root.geometry('600x400')
    app = ImageClassifierApp(root, "Room Messiness Classification App")
