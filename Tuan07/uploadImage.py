

import streamlit as st
import cv2
import numpy as np

loaded_model = None
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

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Run the forward pass to get output of the output layers.
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process(input_image, outputs, classes):
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
        box = boxes[i]  # Sửa chỉ số ở đây
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Draw bounding box.
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        # Class label.
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        # Draw label.
        draw_label(input_image, label, left, top)
    return input_image

def load_model_once():
    global loaded_model
    if loaded_model is None:
        # Load model và các tài liệu khác một lần
        net = cv2.dnn.readNet('D:\\learning\\XLA\\Tuan07\\trai_cay.onnx')

        # Load class names
        classes = []
        with open("D:\\learning\\XLA\\Tuan07\\trai_cay.names", 'r') as f:
            classes = f.read().strip().split('\n')

        loaded_model = (net, classes)

def main():
    st.title("Nhận dạng trái cây")

    # Load model một lần duy nhất
    load_model_once()

    # Cho phép người dùng tải lên một hình ảnh
    uploaded_images = st.file_uploader("Chọn một hình ảnh", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    for uploaded_image in uploaded_images:
        image_data = uploaded_image.read()
        # Xử lý image_data và hiển thị hình ảnh

        # Chuyển dữ liệu hình ảnh thành mảng numpy
        np_image = np.array(bytearray(image_data), dtype=np.uint8)

        # Đọc hình ảnh bằng OpenCV
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Hiển thị hình ảnh
        st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)

        # Unpack loaded model
        net, classes = loaded_model

        # Process image and perform object detection
        outputs = pre_process(image, net)
        detected_image = post_process(image.copy(), outputs, classes)

        # Display the detected image
        st.image(detected_image, caption="Kết quả nhận dạng", use_column_width=True)

if __name__ == "__main__":
    main()
