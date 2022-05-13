import cv2
from src import FaceDetector
from PIL import Image
import numpy as np

detector = FaceDetector()

def camera_detect():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()

        # 将 OpenCV 格式的图片转换为 PIL.Image，注意 PIL 图片是 (width, height)
        pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 绘制带人脸框的标注图
        drawed_pil_im = detector.draw_bboxes(pil_im)
        # 再转回 OpenCV 格式用于视频显示
        frame = cv2.cvtColor(np.asarray(drawed_pil_im), cv2.COLOR_RGB2BGR)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_detect()