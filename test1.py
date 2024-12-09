import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11s.pt')  # 假设你已经下载了YOLO11s的预训练模型

# 加载图片
image_path = 'media/beauty.jpg'
image = cv2.imread(image_path)

# 进行目标检测
results = model(image)

# 可视化检测结果
annotated_image = results[0].plot()

# 显示结果
cv2.imshow('YOLOv8 Inference', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
