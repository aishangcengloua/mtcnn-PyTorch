import numpy as np
from PIL import Image
import torch


def try_gpu():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")

def nms(boxes, overlap_threshold=0.5, mode="union"):
    """非极大值抑制
    Args:
        boxes: 形状为 [n, 5] 的浮点 numpy 数组， 其中每一行是 (xmin, ymin, xmax, ymax, score)
        overlap_threshold: 浮点数，阈值
        mode: 'union' 或者 'min'，计算重叠的指标

    Returns: 保留符合条件的候选框索引的列表
    """
    # 如果没有框，则返回空列表
    if len(boxes) == 0:
        return []

    # 保留框的索引列表
    pick = []

    # 获取边界框的坐标
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # 由小到大

    while len(ids) > 0:

        # 获取最大分数的索引
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        #获得相交矩形的左上角
        x11 = np.maximum(x1[i], x1[ids[ : last]])
        y11 = np.maximum(y1[i], y1[ids[ : last]])
        # 获得相交矩形的右下角角
        x22 = np.minimum(x2[i], x2[ids[ : last]])
        y22 = np.minimum(y2[i], y2[ids[ : last]])

        # 相交区域的宽高
        w = np.maximum(0.0, x22 - x11 + 1.0)
        h = np.maximum(0.0, y22 - y11 + 1.0)

        # 相交面积
        inter = w * h
        if mode == "min":
            overlap = inter / np.minimum(area[i], area[ids[ : last]])
        elif mode == "union":
            # 计算 intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[ : last]] - inter)

        # 将交并比大于阈值的框删除
        left = np.where(overlap < overlap_threshold)
        ids = ids[left]
        # ids = np.delete(
        #     ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        # )

    return pick


def convert_to_square(bboxes):
    """将边界框转换为正方形
    Args:
        bboxes: 形状为 [n, 5] 的浮点 numpy 数组

    Returns: 形状为 [n, 5] 的浮点 numpy 数组， 方形边界框
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0

    # 选宽和高之间的最大值作为正方形边框的边长
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """将边界框转换为更像真正的边界框。 'offsets' 是网络的输出之一
    Args:
        bboxes: 形状为 [n, 5] 的浮点 numpy 数组
        offsets: 形状为 [n, 4] 的浮点 numpy 数组

    Returns: 形状为 [n, 5] 的浮点 numpy 数组
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # 偏移量和预测框、真正边界框的关系:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    """根据边界框得到人脸图像以作为 rnet 的输入
    Args:
        bounding_boxes: 形状为 [n, 5] 的浮点 numpy 数组
        img: PIL.Image
        size: 输出图像的大小

    Returns: 形状为 [n, 3, size, size] 的浮点 numpy 数组
    """
    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(
        bounding_boxes, width, height
    )
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")

        img_array = np.asarray(img, "uint8")
        img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = img_array[
            y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :
        ]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, "float32")

        img_boxes[i, :, :, :] = preprocess(img_box)

    return img_boxes


def correct_bboxes(bboxes, width, height):
    """对一些不符合图像大小的边界框进行处理
    Args:
        bboxes: 形状为 [n, 5] 的浮点 numpy 数组
        width: 原图的宽
        height: 原图的高

    Returns:
        y, x, ey, ex: 形状为 [n] 的 int numpy 数组， 修正 ymin, xmin, ymax, xmax，边界框在原图中的位置
        dy, dx, edy, edx: 原图在边界框中的位置
        h, w: 边界框的高、宽
        返回 [dy, edy, dx, edx, y, ey, x, ex, w, h]
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # 需要从图像中剪下一个框
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # 框的右下角太靠右
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # 框的右下角太低
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # 框的左上角太靠左
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # 框的左上角太靠上
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype("int32") for i in return_list]

    return return_list


def preprocess(img):
    """图像进入网络之前的预处理步骤
    Args:
        img: 形状为 [h, w, c] 的浮点 numpy 数组

    Returns: 形状为 [1, c, h, w] 的浮点 numpy 数组
    """
    img = img.transpose((2, 0, 1))
    # 在数组的相应的axis轴上扩展维度
    img = np.expand_dims(img, 0)
    # 将255的RGB图像归一化到了-1,1的区间，归一化操作，加快收敛速度。
    img = (img - 127.5) * 0.0078125
    return img

# 开始时每一个像素都被当作人脸
def generate_bboxes(probs, offsets, scale, threshold):
    """在可能有脸的地方生成边界框
    Args:
        probs: 形状为 [n, m] 的浮点 numpy 数组
        offsets: 形状为 [1, 4, n, m] 的浮点 numpy 数组
        scale: 浮点数，图像的宽度和高度的缩放缩放比例
        threshold: 该方格为人脸的概率

    Returns: 形状为 [n_boxes, 9] 的浮点 numpy 数组

    """
    # 在某种意义上，P-Net 的结果可以由一个大小为 (12, 12) 的移动窗口以步长为 2 进行卷积得到
    stride = 2
    cell_size = 12

    # 可能有脸的方格的索引
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # 边界框的变换
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # 这里主要是求 x1,x2,y1,y2
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net 应用于缩放图像，所以我们需要重新缩放边界框
    bounding_boxes = np.vstack(
        [
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score,
            offsets,
        ]
    )

    return bounding_boxes.T