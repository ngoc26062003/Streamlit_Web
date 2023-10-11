import sys
import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs
import cv2
import numpy as np

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

class Main(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
        self.classes = None
        self.getclass()
        self.net = cv2.dnn.readNet('trai_cay.onnx')
  
    def getclass(self):
        classesFile = "trai_cay.names"
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def initUI(self):
        self.parent.title("Phat Hien Trai Cay - Yolo5")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar, tearoff=False)
        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_command(label="Recognition", command=self.onRecognition)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)

        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)
  
    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_COLOR)
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ImageIn", imgin)

    def draw_label(self, im, label, x, y):
        """Draw text onto image at location."""
        # Get text size.
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle.
        cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
        # Display text inside the rectangle.
        cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

    def pre_process(self, input_image):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

        # Sets the input to the network.
        self.net.setInput(blob)

        # Run the forward pass to get output of the output layers.
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs

    def post_process(self, input_image, outputs):
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        # Rows.
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        # Resizing factor.
        x_factor = image_width / INPUT_WIDTH
        y_factor =  image_height / INPUT_HEIGHT
        # Iterate through detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                    classes_scores = row[5:]
                    # Get the index of max class score.
                    class_id = np.argmax(classes_scores)
                    #  Continue if the class score is above threshold.
                    if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)

        # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # Draw bounding box.             
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            # Class label.                      
            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])             
            # Draw label.             
            self.draw_label(input_image, label, left, top)
        return input_image

    def onRecognition(self):
        global imgout
        # Process image.
        detections = self.pre_process(imgin)
        imgout = self.post_process(imgin.copy(), detections)
        """
        Put efficiency information. The function getPerfProfile returns       the overall time for inference(t) 
        and the timings for each of the layers(in layersTimes).
        """
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
        print(label)
        #cv2.putText(imgout, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)


root = Tk()
Main(root)
root.geometry("640x480+100+100")
root.mainloop()

