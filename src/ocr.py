import easyocr
import cv2
import numpy as np
import os
from pathlib import Path

reader = easyocr.Reader(['en'], gpu=True)

def preprocess_image(img, apply_contrast=True, contrast_factor=1.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if apply_contrast:
        gray = cv2.convertScaleAbs(gray, alpha=contrast_factor, beta=0)
    
    
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    threshold = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    return threshold


def extract_raw_text(image_path, apply_preprocessing=True, contrast_factor=1.5, 
                     text_threshold=0.5, save_debug=True, print_confidence=True):
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not read image at {image_path}")
            return ""
        
        if apply_preprocessing:
            processed_img = preprocess_image(img, apply_contrast=True, contrast_factor=contrast_factor)
        else:
            processed_img = img
        
        if save_debug:
            debug_dir = Path("extracted/raw_text")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            debug_path = debug_dir / f"{image_name}_preprocessed.png"
            cv2.imwrite(str(debug_path), processed_img)
        
        result = reader.readtext(processed_img, detail=1)
        
        filtered_results = []
        for detection in result:
            bbox, text, confidence = detection
            
            if confidence >= text_threshold:
                filtered_results.append((text, confidence))
                
                if print_confidence:
                    print(f"Text: '{text}' | Confidence: {confidence:.3f}")
            else:
                if print_confidence:
                    print(f"Skipped (low confidence): '{text}' | Confidence: {confidence:.3f}")

        if not filtered_results:
            print(f"Warning: No text detected above threshold {text_threshold} in {image_path}")
            return ""
        
        extracted_text = " ".join([text for text, _ in filtered_results])
    
        if save_debug:
            text_output_path = debug_dir / f"{image_name}_output.txt"
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Text threshold: {text_threshold}\n")
                f.write(f"Preprocessing: {apply_preprocessing}\n")
                f.write(f"Contrast factor: {contrast_factor}\n")
                f.write("-" * 50 + "\n")
                for text, conf in filtered_results:
                    f.write(f"{text} (confidence: {conf:.3f})\n")
                f.write("-" * 50 + "\n")
                f.write(f"Combined text:\n{extracted_text}\n")
        
        return extracted_text
    
    except Exception as e:
        print(f"Error during OCR extraction from {image_path}: {str(e)}")
        return ""
