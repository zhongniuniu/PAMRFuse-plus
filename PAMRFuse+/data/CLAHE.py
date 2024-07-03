from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


test = Image.open('/public2/zhongyutian/Liuzhenyang/OCT/testB/fangzhen.jpg').convert('L')
test = np.uint8(test)

test_hist = cv2.equalizeHist(test)
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(10,5))
test_clahe = clahe.apply(test)

cv2.imwrite('/public2/zhongyutian/Liuzhenyang/Clahe_result/fangzhen.jpg', test_clahe)
# plt.figure()
# plt.subplot(1,3,1),plt.imshow(test, 'gray')
# plt.axis('off'),plt.title('原图')
# plt.subplot(1,3,2),plt.imshow(test_hist, 'gray')
# plt.axis('off'),plt.title('直方图均衡化')
# plt.subplot(1,3,3),plt.imshow(test_clahe, 'gray')
# plt.axis('off'),plt.title('Clahe')
# plt.show()
