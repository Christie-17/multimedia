import base64
import os
import cv2
import requests
from aip import AipOcr  # 百度AI的文字识别库
import matplotlib.pyplot as plt
import time
import queue
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
path1 = './video/material/%s.jpg'  # 视频转为图片存放的路径（帧）
path2 = './video/img/%s.jpg'  # 图片经过边缘提取后存放的路径
path3 = './video/captions/%s.jpg'  # 图片经过边缘提取后存放的路径

begin = 10
end = 100
step_size = 10
videoName = "material"


def tailor_video():
    # 在这里把后缀接上
    video_path = os.path.join("D:/project1/", videoName + '.mp4')
    times = 0
    frameFrequency = 10  # 提取视频的频率，每10帧提取一个
    outPutDirName = './video/' + videoName + '/'
    if not os.path.exists(outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(outPutDirName)
    camera = cv2.VideoCapture(video_path)
    while True:
        times += 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            cv2.imwrite(outPutDirName + str(times) + '.jpg', image)  # 文件目录下将输出的图片名字命名为10.jpg这种形式
            print(outPutDirName + str(times) + '.jpg')
    print('图片提取结束')


def view_img(img):
    cv2.imshow("img", img)
    cv2.waitKey()


Q = queue.Queue()

dx = [0, 1, 0, -1, 1, -1, 1, -1]
dy = [1, 0, -1, 0, -1, 1, 1, -1]

vis = None


class POS:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def BFS(img, x, y, flag):
    global vis
    theta_low = 1
    #theta_high = 50
    theta_high = 500
    n = img.shape[0]
    m = img.shape[1]
    Q.put(POS(x, y))
    vis[x][y] = 1
    min_x = x
    max_x = x
    min_y = y
    max_y = y
    while not Q.empty():
        top = Q.get()
        for i in range(8):
            X = dx[i] + top.x
            Y = dy[i] + top.y
            if X < 0 or X >= n or Y < 0 or Y >= m or img[X][Y]==0:
                continue
            if flag:
                img[X][Y] = 0
            elif vis[X][Y]:
                continue
            min_x = min(min_x, X)
            max_x = max(max_x, X)
            min_y = min(min_y, Y)
            max_y = max(max_y, Y)
            vis[X][Y] = 1
            tmp = POS(X, Y)
            Q.put(tmp)
    len_x = (max_x - min_x + 1)
    len_y = (max_y - min_y + 1)
    """
    if not flag:
        print(len_x, len_y)
    """
    """
    ret = True
    if len_x<=theta_low and len_y<=theta_low:
        ret=False
    if len_x>=theta_high and len_y>=theta_high:
        ret=False
    if len_x<=theta_low and len_y<=theta_high and
    """
    ret = len_x >= theta_low and len_x <= theta_high and len_y >= theta_low and len_y <= theta_high

    return ret


def size_filter(img):
    # np.set_printoptions(threshold=np.inf)
    # print(img)
    n = img.shape[0]
    m = img.shape[1]
    global vis
    vis = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if img[i][j] > 0:
                flag = BFS(img, i, j, False)
                if not flag:
                    img[i][j] = 0
                    BFS(img, i, j, True)
    #view_img(img)


def static(img):
    theta = 40
    n = img.shape[0]
    m = img.shape[1]
    row = np.zeros(n)
    column = np.zeros(m)
    # 统计区域像素
    for i in range(n):
        for j in range(m):
            x = img[i][j]
            if x > theta:
                row[i] += 1
                column[j] += 1

    # 求灰度比例
    row /= m
    column /= n

    # 根据阈值分段
    row = row > 0.08
    column = column > 0.005

    for i in range(n):
        for j in range(m):
            if img[i][j]>theta:
                img[i][j] *= (row[i] and column[j])
            else:
                img[i][j]=0

    #view_img(img)


def abstract():  # 截取字幕
    for i in range(begin, end, step_size):
        fname1 = path1 % str(i)
        print(fname1)
        img = cv2.imread(fname1)
        if img is None:
            break;
        print(img.shape)
        cropped = img
        """
        imgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        thresh = 200
        ret, binary = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)  # 输入灰度图，输出二值图
        binary1 = cv2.bitwise_not(binary)  # 取反
        cv2.imwrite(path2 % str(i), binary1)
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """
        # sobel边缘检测
        edges = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        # 浮点型转成uint8型
        edges = cv2.convertScaleAbs(edges)
        """
        img = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

        size_filter(img)

        static(img)
        #view_img(img)
        #thresh=100
        #_,img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)  # 输入灰度图，输出二值图
        cv2.imwrite(path3 % str(i), img)

# 定义一个函数，用来访问百度API，
def requestApi(img):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    params = {"image": img,'language_type':'CHN_ENG'}
    #access_token = '24.125c4d691b4c97c080b647f0d07c5419.2592000.1634915041.282335-24887036'
    access_token='24.8b5758ec70247b64ccb608d52e720c1c.2592000.1634969842.282335-24889932'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    results=response.json()
    return results

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        # 将读取出来的图片转换为b64encode编码格式
        return base64.b64encode(fp.read())

def subtitle():
    for i in range(begin,end,step_size):
        time.sleep(0.5)  # api-qpr=2
        fname1=path3 % str(i)
        print(fname1)
        image = get_file_content(fname1)
        try:
            #print(requestApi(image))
            ret = requestApi(image)
            #print(ret)
            results=ret['words_result']
            if ret.get('error_msg'):
                print(ret['error_msg'])
            for item in results:
                print(item['words'])
        except Exception as e:
            print(e)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #abstract()
    subtitle()
    """
    
    reader = easyocr.Reader(['ch_sim', 'en'])
    result = reader.readtext('./video/captions2/10.jpg')
    """