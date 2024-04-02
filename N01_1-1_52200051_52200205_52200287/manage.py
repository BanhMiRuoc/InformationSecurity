import streamlit as st
from deta import Deta
import cv2 as cv
import numpy as np
import os

DETA_KEY = 'c03et5kylxk_UTfpW64PgHBWrD5Eg9sc7jZaqArPWDtq'
deta = Deta(DETA_KEY)
db = deta.Base('StreamlitAuth')

def insert_user(username, password, path):
    return db.put({'username': username, 'password': password, 'img': path})

def fetch_users() -> list:
    users = db.fetch()
    return users.items

def getUsernames() -> list:
    users = db.fetch()
    return [x['username'] for x in users.items]

def sign_up():
    st.title("Đăng ký")
    with st.form("regist"):
        username = st.text_input("Tài khoản")
        password = st.text_input("Mật khẩu", type="password")
        confirm_pw = st.text_input("Xác nhận mật khẩu", type="password")
        submit = st.form_submit_button('Đăng ký')

    if submit:
        if username not in getUsernames():
            file1 = input_picture()
            st.subheader("Nhấn ESC trên bàn phím để chụp ảnh mống mắt")
            if password == confirm_pw:
                insert_user(username, confirm_pw, file1)
                st.success("Đăng ký thành công!")
                st.balloons()
            else:
                st.warning("Mật khẩu không khớp")
        else:
            st.warning("Tài khoản đã tồn tại!")

def sign_in():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    users = fetch_users()
    listUser = [user for user in users]
    
    st.title("Đăng nhập")
    with st.form("login"):
        username = st.text_input("Tên tài khoản")
        submit = st.form_submit_button('Đăng nhập')
    
    if submit:
        if username is None or username == "":
            st.error("Vui lòng nhập username.")
            st.stop()
        user = None
        for x in listUser:
            if x['username'] == username:
                user = x
                break
        if user is None:
            st.error("Tài khoản không tồn tại.")
            st.stop()
        st.subheader("Nhấn ESC trên bàn phím để chụp ảnh mống mắt")

        tmp_path_input_img = input_for_compare()
        current_directory = os.path.dirname(os.path.abspath(__file__))
        path_input_img = os.path.join(current_directory, tmp_path_input_img)

        if compare(os.path.join(current_directory, user['img']), path_input_img):
            st.success("Đăng nhập thành công")
            st.header("Đã demo xong, cảm ơn cô đã xem!")
        else:
            st.warning("Ảnh mống mắt không phù hợp")
        
def input_picture():
    cap = cv.VideoCapture(0)
    while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            cv.imshow('img', frame)
            file = "input.jpg"
            cv.imwrite(file, frame)
            key = cv.waitKey(1) & 0xFF
            if key == 27:
                break
    cap.release()
    cv.destroyAllWindows()
    return file

def input_for_compare():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        cv.imshow('img', frame)
        file = "input1.jpg"
        cv.imwrite(file, frame)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cv.destroyAllWindows()
    return file

def compare(img_path1, img_path2):
    
    def detect_eyes(image_path):
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        left_eye = ()
        right_eye = ()
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = img[y: y + h, x: x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                if ex < w/2:
                    left_eye = (ex, ey, ew, eh)
                else:
                    right_eye = (ex, ey, ew, eh)
            
            if left_eye:
                x_left, y_left, w_left, h_left = left_eye
                left_eye_image = roi_color[y_left: y_left + h_left, x_left: x_left + w_left]

            if right_eye:
                x_right, y_right, w_right, h_right = right_eye
                right_eye_image = roi_color[y_right: y_right + h_right, x_right: x_right + w_right]
                
            return left_eye_image, right_eye_image
        
    def normalize_eye(eye_image):
        if len(eye_image.shape) > 2:
            eye_image = cv.cvtColor(eye_image, cv.COLOR_BGR2GRAY)
        return cv.equalizeHist(eye_image)

    def compare_eyes(eye_image1, eye_image2):
        gray_eye1 = normalize_eye(eye_image1)
        gray_eye2 = normalize_eye(eye_image2)
        orb = cv.ORB_create()

        keypoints1, descriptors1 = orb.detectAndCompute(gray_eye1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray_eye2, None)

        if descriptors1 is None or descriptors2 is None:
            print("Không thể tìm thấy descriptors cho mống mắt.")
            return False

        descriptors1, descriptors2 = descriptors1.astype(np.float32), descriptors2.astype(np.float32)

        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        return len(sorted(matches, key=lambda x: x.distance))

    l_eye_img1, r_eye_img1 = detect_eyes(img_path1)
    l_eye_img2, r_eye_img2 = detect_eyes(img_path2)
    if ((compare_eyes(l_eye_img1, l_eye_img2) >= 5) or (compare_eyes(r_eye_img1, r_eye_img2) >= 5)):
        return True
    return False
    
if __name__ == "__main__":
    st.title(':blue[IRIS RECOGNITION]')
    tabs = {"Đăng nhập": sign_in, "Đăng ký": sign_up}
    choice = st.sidebar.selectbox("Chọn chức năng", tabs)
    tabs[choice]()

    