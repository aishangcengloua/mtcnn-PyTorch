import math
from PIL import ImageDraw
from torch.autograd import Variable
from .model import PNet, RNet, ONet
from .utils import *

class FaceDetector:
    def __init__(self, device=try_gpu()):
        self.device = device

        # 下载模型参数
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)

        # 如果模型中有BN层(BatchNormalization）和Dropout，在测试时添加model.eval()。
        # model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
        # 对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

        self.onet.eval()

    def detect(self, image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        """
        Args:
            image: PIL.Image 类型
            min_face_size: 浮点数，关注的人脸的最小大小
            thresholds: 长度为 3 的列表，初次筛选框的阈值，即是否为人脸的阈值
            nms_thresholds: 长度为 3 的列表，非极大值抑制的阈值

        Returns:
            bounding_boxes: 形状为 [n_boxes, 5]，[左上角x坐标偏移量, 左上角y坐标偏移量, 右下角x坐标偏移量, 右下角y坐标偏移量, 检测评分]
            landmarks: 形状为[n_boxes, 10]，[右眼x偏移量, 左眼x偏移量, 鼻子x偏移量, 右嘴角x偏移量, 左嘴角x偏移量,
                                           右眼y偏移量, 左眼y偏移量, 鼻子y偏移量, 右嘴角y偏移量, 左嘴角y偏移量]
        """
        # 建立图像金字塔
        width, height = image.size
        min_length = min(height, width)

        # P - Net用于检测12×12大小的人脸，注意并不是图片的真正大小
        # 原图中小于 min_face_size 这个尺寸的人脸不必 care
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # 缩放比例
        scales = []

        #缩放图像以便我们可以检测到的最小尺寸等于我们想要检测的最小人脸尺寸

        # 图像缩小比（为缩放尺寸的倒数）
        # 最大缩放尺度，最小缩小尺度max_scale = min_detection_size / min_face_size = 12 / 20，
        # 最小缩放尺度，最大缩小尺度min_scale = min_detection_size / max_face_size = 12 / 100，
        # 中间的缩放尺度scale_n = max_scale * (factor ^ n)

        # 原图中 min_face_size 的人脸变成了 12 * 12，即最大缩放尺度，最小缩小尺度
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # 预测框
        bounding_boxes = []

        # 使用不同缩放比例的图像进行 PNet 预测
        for s in scales:
            boxes = self.__run_first_stage(image, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # 从不同的尺度收集框（以及偏移量和分数）
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # 使用 pnet 预测的偏移量来变换边界框
        # offset 有归一化过程， 因此要返归一化操作
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes).to(self.device))
            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes).to(self.device))
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # 计算地标
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]

        # lankmark 返回的是从左到右，从上到下的长度的比例
        landmarks[:, 0:5] = (
            np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        )

        landmarks[:, 5:10] = (
            np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        )

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    def draw_bboxes(self, image):
        """绘制边界框和面部地标
        Args:
            image: PIL.Image

        Returns: PIL.Image
        """
        bounding_boxes, facial_landmarks = self.detect(image)

        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        for b in bounding_boxes:
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="white", width=3)
            # score = float(re.findall(r"\d{1,}?\.\d{2}", str(b[4]))[0])
            # draw.text((b[0], b[1]), f'{score}')


        for p in facial_landmarks:
            for i in range(5):
                draw.ellipse(
                    [(p[i] - 1.0, p[i + 5] - 1.0), (p[i] + 1.0, p[i + 5] + 1.0)],
                    outline="darkblue", width=2)

        return img_copy

    def crop_faces(self, image, size=112):
        """裁剪所有面部图像
        Args:
            image: PIL.Image
            size: 输出图像的边长

        Returns:
        """
        bounding_boxes, _ = self.detect(image)
        img_list = []

        # 将边界框转成正方形
        square_bboxes = convert_to_square(bounding_boxes)

        for b in square_bboxes:
            face_img = image.crop((b[0], b[1], b[2], b[3]))
            face_img = face_img.resize((size, size), Image.BILINEAR)
            img_list.append(face_img)
        return img_list

    def __run_first_stage(self, image, scale, threshold):
        """运行 PNet，生成边界框，并进行 NMS
        Args:
            image: PIL.Image
            scale: 浮点数， 按此数字缩放图像的宽度和高度
            threshold: 浮点数，来自网络预测的边界框是人脸概率的阈值

        Returns: 形状为 [n_boxes, 9] 的浮点 numpy 数组， 带有分数和偏移量的边界框 (4 + 1 + 4)
        """
        # 缩放图像并将其转换为浮点数组
        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, "float32")

        with torch.no_grad():
            #可以不用求导
            img = Variable(torch.FloatTensor(preprocess(img)).to(self.device))
            output = self.pnet(img)
            # print(output)
            # 取第二个特征图的结果作为滑动窗口的非人脸概率
            # 取第二个特征图的结果作为滑动窗口的人脸概率
            probs = np.array(output[1].cpu().data.numpy()[0, 1, :, :])
            offsets = output[0].cpu().data.numpy()

            # probs: 每个方格为人脸的概率
            # offsets: 预测框对真正框的偏移量

        # 根据偏移量和缩放比例返回矩形框在原始图像的位置所在
        boxes = generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]