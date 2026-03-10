"""
Inference pipeline for Pneumonia Detection.
Provides easy-to-use interface for making predictions on new images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import (
    IMAGE_SIZE, MEAN, STD, CLASSES, BEST_MODEL_PATH, get_device
)
from src.model import PneumoniaModel, PneumoniaModelEfficientNet
from src.utils import load_checkpoint
from src.visualization import GradCAMVisualizer, denormalize_image


class PneumoniaPredictor:
    """
    Predictor class for pneumonia detection.
    Handles model loading, image preprocessing, and prediction.
    """
    
    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        device: str = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
        """
        self.device = device if device else get_device()
        self.model_path = model_path
        self.classes = CLASSES
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize Grad-CAM visualizer
        self.gradcam = GradCAMVisualizer(self.model)
        
        print(f"Predictor initialized on {self.device}")
    
    def _load_model(self):
        """Load the trained model."""
        # Try EfficientNet first (new default), fallback to ResNet50
        try:
            model = PneumoniaModelEfficientNet(num_classes=len(self.classes))
            if self.model_path.exists():
                load_checkpoint(self.model_path, model, device=self.device)
                print(f"EfficientNet model loaded from: {self.model_path}")
            return model.to(self.device)
        except RuntimeError:
            # If EfficientNet fails, try ResNet50
            print(f"EfficientNet load failed, trying ResNet50...")
            model = PneumoniaModel(num_classes=len(self.classes))
            if self.model_path.exists():
                load_checkpoint(self.model_path, model, device=self.device)
                print(f"ResNet50 model loaded from: {self.model_path}")
            else:
                print(f"Warning: Model not found at {self.model_path}")
                print("Using uninitialized model!")
            return model.to(self.device)
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for prediction.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Resize
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array(MEAN)
        std = np.array(STD)
        img_array = (img_array - mean) / std
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_probabilities: bool = False
    ) -> Union[str, Tuple[str, float], dict]:
        """
        Make prediction on an image.
        
        Args:
            image: Input image
            return_probabilities: Whether to return all class probabilities
            
        Returns:
            Prediction result (class name or dict with probabilities)
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        probs = probabilities.cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        result = {
            'class': self.classes[predicted_class],
            'confidence': float(confidence),
            'class_index': int(predicted_class),
            'probabilities': {
                cls: float(prob) for cls, prob in zip(self.classes, probs)
            }
        }
        
        if return_probabilities:
            return result
        else:
            return result['class'], result['confidence']
    
    def predict_with_gradcam(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> dict:
        """
        Make prediction with Grad-CAM visualization.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing prediction and Grad-CAM heatmap
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        probs = probabilities.cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        # Generate Grad-CAM
        cam = self.gradcam.generate_cam(img_tensor)
        
        # Get original image for visualization
        if isinstance(image, (str, Path)):
            orig_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            orig_image = Image.fromarray(image).convert('RGB')
        else:
            orig_image = image
        
        orig_image = orig_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        orig_array = np.array(orig_image)
        
        # Create overlay
        overlay = self.gradcam.visualize(orig_array, cam)
        
        return {
            'class': self.classes[predicted_class],
            'confidence': float(confidence),
            'class_index': int(predicted_class),
            'probabilities': {
                cls: float(prob) for cls, prob in zip(self.classes, probs)
            },
            'cam': cam,
            'overlay': overlay,
            'original_image': orig_array
        }
    
    def predict_batch(
        self,
        images: list,
        return_probabilities: bool = False
    ) -> list:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of input images
            return_probabilities: Whether to return all class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict(image, return_probabilities=return_probabilities)
            results.append(result)
        return results


def predict_single_image(
    image_path: Union[str, Path],
    model_path: Path = BEST_MODEL_PATH,
    show_gradcam: bool = False
) -> dict:
    """
    Convenience function to predict on a single image.
    
    Args:
        image_path: Path to the image
        model_path: Path to the model checkpoint
        show_gradcam: Whether to generate Grad-CAM visualization
        
    Returns:
        Prediction result dictionary
    """
    predictor = PneumoniaPredictor(model_path=model_path)
    
    if show_gradcam:
        result = predictor.predict_with_gradcam(image_path)
    else:
        result = predictor.predict(image_path, return_probabilities=True)
    
    # Print results
    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)
    print(f"Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nProbabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.2%}")
    print("=" * 50)
    
    return result


def main():
    """Main function for testing predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray')
    parser.add_argument('image', type=str, help='Path to the X-ray image')
    parser.add_argument('--model', type=str, default=str(BEST_MODEL_PATH),
                       help='Path to model checkpoint')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate Grad-CAM visualization')
    
    args = parser.parse_args()
    
    # Run prediction
    result = predict_single_image(
        args.image,
        model_path=Path(args.model),
        show_gradcam=args.gradcam
    )
    
    # Save Grad-CAM if requested
    if args.gradcam and 'overlay' in result:
        output_path = Path(args.image).stem + '_gradcam.png'
        cv2.imwrite(str(output_path), cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
        print(f"\nGrad-CAM saved to: {output_path}")


if __name__ == "__main__":
    main()
