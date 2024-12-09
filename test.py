import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11s.pt')  # 假设你已经下载了YOLOv8的预训练模型

# 打开视频文件
video_path = 'media/cars.mp4'
cap = cv2.VideoCapture(video_path)

# 处理视频帧
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 进行目标检测
        results = model(frame)
        
        # 可视化检测结果
        annotated_frame = results[0].plot()
        
        # 显示结果
        cv2.imshow('YOLOv8 Inference', annotated_frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
