from src import FaceDetector
from PIL import Image

# 人脸检测对象。优先使用GPU进行计算
detector = FaceDetector()

image = Image.open("./images/face_1.jpg")

# 检测人脸，返回人脸位置坐标
bboxes, landmarks = detector.detect(image)

# 绘制并保存标注图
drawed_image = detector.draw_bboxes(image)
drawed_image.save("./images/drawed_image.jpg")
drawed_image.show()

# 裁剪人脸图片并保存
# face_img_list = detector.crop_faces(image, size=64)
# for i in range(len(face_img_list)):
#     face_img_list[i].save("./images/face_" + str(i + 1) + ".jpg")