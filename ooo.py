import cv2
import numpy as np

# 1. إعداد الكاميرا (هنا العميل هيقدر يحط لينك الـ RTSP بسهولة)
cam = cv2.VideoCapture(0)

cam.set(3, 740)
cam.set(4, 580)

# 2. تحميل أسماء الأشياءqq
class_file = 'things.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# 3. إعداد الموديل
p = 'frozen_inference_graph.pb'
v = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v)
net.setInputSize(320, 320)  # يفضل يكون مربع للأداء الأفضل مع SSD
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cam.read()
    if not success:
        break

    # 4. كشف الأشياء مع رفع نسبة الثقة (confThreshold) لـ 0.6 لتقليل الأخطاء
    classIds, confs, bbox = net.detect(img, confThreshold=0.6)

    # تحويل النتائج لـ لستة عشان نطبق الـ NMS
    if len(classIds) != 0:
        indices = cv2.dnn.NMSBoxes(bbox, confs, score_threshold=0.6, nms_threshold=0.4)

        for i in indices:
            # i هنا هو رقم المربع اللي فاز في التصفية
            box = bbox[i]
            classId = classIds[i]
            confidence = confs[i]
            label = class_names[classId - 1].upper()

            # 5. الفلتر السحري: اظهر "PERSON" بس (طلب العميل)
            if label == "PERSON":
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                # عرض اسم الكائن ونسبة التأكد
                text = f"{label} {int(confidence * 100)}%"
                cv2.putText(img, text, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Output', img)
    k=cv2.waitKey(1)
    if k == ord('q'):
        cv2.destroyAllWindows()


