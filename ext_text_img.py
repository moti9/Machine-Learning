import cv2
import pytesseract

# Load the image
image_path = "IELTS-template.jpg"
img = cv2.imread(image_path)

# Preprocess the image (optional, based on image characteristics)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), 0)

# Additional preprocessing steps (example)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Extract text using OCR with additional configurations
# custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode (OEM) 3 and Page Segmentation Mode (PSM) 6
text = pytesseract.image_to_string(img)

# Display the result
print("Extracted Text:", text)
