import cv2
import numpy as np
import time
import argparse

prototxt_path='src/deploy.prototxt.txt'
caffemodel_path='src/weights.caffemodel'
CONFIDENCE=0.5

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def face_pixelate(image,blocks=3):
    (h,w)=image.shape[:2]
    Xsteps=np.linspace(0,w,blocks+1,dtype='int')
    Ysteps=np.linspace(0,h,blocks+1,dtype='int')
    for i in range(1,len(Ysteps)):
        for j in range(1,len(Xsteps)):
            startX=Xsteps[j-1]
            endX=Xsteps[j]
            startY=Ysteps[i-1]
            endY=Ysteps[i]
            roi=image[startY:endY,startX:endX]
            (B,G,R)=[int(x) for x in cv2.mean(roi)[:3]]
            res=cv2.rectangle(image,(startX,startY),(endX,endY),(B,G,R),-1)
    return image

def face_blurring(image,blur_type,blocks=3):
    img_copy=np.copy(image)
    (h, w)=img_copy.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(img_copy,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    model.setInput(blob)
    detections=model.forward()
    for i in range(0,detections.shape[2]):
        confidence=detections[0, 0, i, 2]
        if confidence>CONFIDENCE:
            box=detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face=img_copy[startY:endY,startX:endX]
            if(blur_type=="pixelate"):
                face=face_pixelate(face,blocks)
            else:
                face=cv2.blur(face,(23, 23))
                img_copy[startY:endY,startX:endX]=face
    return img_copy

def video_blurring(video,blur_type,blocks,path):
    cap=cv2.VideoCapture(video)
    time.sleep(2.0)
    res,frame=cap.read()
    h,w=frame.shape[:2]
    fourcc=cv2.VideoWriter_fourcc(*"XVID")
    out=cv2.VideoWriter(path,fourcc,20.0,(w,h))
    while True:
        res,frame=cap.read()
        if frame is None:
            break
        output=face_blurring(frame,blur_type,blocks)
        out.write(output)
        #if cv2.waitKey(1) & 0xFF==ord('a'):
        #    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("input", required=True, help="path to input video")
    ap.add_argument("output", required=True, help="path to output video with extension")
    ap.add_argument("blur_type",type=str, default="simple",choices=["simple", "pixelated"], help="face blurring/anonymizing method")
    ap.add_argument("blocks", type=int, default=15, help="# of blocks for the pixelated blurring method")
    args = vars(ap.parse_args())
    video_blurring(ap.video,ap.blur_type,ap.blocks,ap.output)
