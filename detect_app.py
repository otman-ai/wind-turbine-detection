import torch
import cv2
import matplotlib.pyplot as plt
model_path = 'D:/Path to Successed/End to End Projects/Turbines Detection/model/best.pt'
device = 'cpu'  # replace with 'cuda' if you have a GPU

# Load the YOLOv5 model
model = torch.hub.load('D:/Path to Successed/End to End Projects/Turbines Detection/yolov5', 'custom', path=model_path, source='local')

# Load the input image
#img = cv2.imread(img_path)
img = cv2.imread("tested_images/wind-farm.jpg")
# Run the YOLOv5 model on the input image
results = model(img)

# Get the detected objects and their bounding boxes
objects = results.pred[0].detach().cpu().numpy()
font_scale = .5
font_thickness = 1
font_color = (0, 255, 0)
# Draw the bounding boxes on the image
for obj in objects:
    x1, y1, x2, y2, conf, cls = [int(x) for x in obj]
    if obj[4]>0.65:
        cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0, 255, 0), 2)
        cv2.putText(img, 'Wind Turbines', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

# Display the image
cv2.imwrite("img.jpg",img)
plt.imshow(img)
plt.show()