import torch
import cv2

# yolov5s.pt load
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2. opencv opencv 로 이미지를 읽고 이미지의 가로, 세로가 각 몇 pixel 인지 구하세요
tmp_img = cv2.imread('people.jpeg')
print(tmp_img.shape) # (837, 1024, 3)

results = model(tmp_img)
print(results.pandas().xyxy[0])
result = results.pandas().xyxy[0].to_numpy() # 판다스 dataframe을 numpy로 변환

result = [item for item in result if item[6]=='person'] # person인 데이터만 추출
print(result)
for i, person in enumerate(result):
    i += 1
    x_min = int(person[0])
    y_min = int(person[1])
    x_max = int(person[2])
    y_max = int(person[3])
    
    cropped = tmp_img[y_min:y_max, x_min:x_max]
    cv2.imwrite(f'people{i}.png', cropped)
    
    cv2.rectangle(tmp_img, (x_min, y_min), (x_max, y_max), (255, 255, 255))

cv2.imwrite('result1.png', tmp_img)
