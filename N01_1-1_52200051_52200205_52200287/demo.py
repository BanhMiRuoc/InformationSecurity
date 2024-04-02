import cv2
import numpy as np

def detect_eyes(image_path, path_out1, path_out2):
    # Load bộ lọc Haar Cascade cho việc nhận diện khuôn mặt và mắt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Đọc ảnh
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Với mỗi khuôn mặt, phát hiện mắt
    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Phát hiện mắt trong khuôn mặt
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Lấy vùng mắt trái và vùng mắt phải
        left_eye = []
        right_eye = []

        for (ex, ey, ew, eh) in eyes:
            # Xác định mắt trái và mắt phải
            if ex < w/2:
                left_eye = (ex, ey, ew, eh)
            else:
                right_eye = (ex, ey, ew, eh)

        # Cắt vùng chứa mắt trái và mắt phải từ ảnh gốc
        if left_eye:
            x_left, y_left, w_left, h_left = left_eye
            left_eye_image = roi_color[y_left:y_left+h_left, x_left:x_left+w_left]
            cv2.imshow('Left Eye', left_eye_image)
            cv2.imwrite(path_out1, left_eye_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if right_eye:   
            x_right, y_right, w_right, h_right = right_eye
            right_eye_image = roi_color[y_right:y_right+h_right, x_right:x_right+w_right]
            cv2.imshow('Right Eye', right_eye_image)
            cv2.imwrite(path_out2, right_eye_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Đường dẫn đến ảnh cần phân tích
image_path = ['input1.jpg', 'input2.jpg']
path_output1 = ['origin_image/eye1/left_eye.jpg', 'origin_image/eye1/right_eye.jpg']
path_output2 = ['origin_image/eye2/left_eye.jpg', 'origin_image/eye2/right_eye.jpg']

# Gọi hàm để phát hiện và cắt vùng mắt trái và mắt phải từ ảnh
detect_eyes(image_path[0], path_output1[0], path_output1[1])
detect_eyes(image_path[1], path_output2[0], path_output2[1])

# ------------------------------------------------------------------------------------------------------------ #

def read_and_convert_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def detect_and_extract_features(image):
    # Example using ORB features
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def compare_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return 0  # Return low similarity if descriptors are not found

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return len(matches)

# Paths to two eye images for testing
image_path1 = 'origin_image/eye1/left_eye.jpg'
image_path2 = 'origin_image/eye2/left_eye.jpg'

# Read and convert images to grayscale
image1 = read_and_convert_image(image_path1)
image2 = read_and_convert_image(image_path2)

# Detect and extract features
keypoints1, descriptors1 = detect_and_extract_features(image1)
keypoints2, descriptors2 = detect_and_extract_features(image2)


# Compare features
# Set a threshold for similarity
threshold = compare_features(descriptors1, descriptors1)  # Adjust as needed
similarity_score = compare_features(descriptors1, descriptors2) / threshold

if similarity_score >= 0.5:
    print(f'The eyes are similar! Similarity Score: {similarity_score}')
else:
    print(f'The eyes are dissimilar. Similarity Score: {similarity_score}')
