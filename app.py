import cv2 
import torch

print("Launch the camera.")
vid = cv2.VideoCapture(0)

#Setup the model path
model_path = 'D:/Path to Successed/End to End Projects/Turbines Detection/model/best.pt'
device = 'cpu'  # replace with 'cuda' if you have a GPU

# Load the YOLOv5 model
print("Loading the model...")
model = torch.hub.load('D:/Path to Successed/End to End Projects/Turbines Detection/yolov5', 'custom', path=model_path, source='local')
print("The model loaded successfuly.")
print("Detecting..")
while(True):
    ret, frame = vid.read()
    results = model(frame)
    objects = results.pred[0].detach().cpu().numpy()
    #Setup font scale,thickness,color
    font_scale = .5
    font_thickness = 1
    font_color = (0, 255, 0)
    # Draw the bounding boxes on the image
    for obj in objects:
        x1, y1, x2, y2, conf, cls = [int(x) for x in obj]
        if obj[4]>0.65:
            #Draw rectangle on detected object
            cv2.rectangle(frame, (x1, y1), (x1+x2, y1+y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Wind Turbines', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        #Show the frame with detected object
        cv2.imshow("Turbines Detection",frame)
    
    # Stop the camera when the use rpress "q" key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()