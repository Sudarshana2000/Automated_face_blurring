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

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("image", required=True, help="path to input image")
    ap.add_argument("output", required=True, help="path to output image")
    ap.add_argument("blur_type",type=str, default="simple",choices=["simple", "pixelated"],
	help="face blurring/anonymizing method")
    ap.add_argument("blocks", type=int, default=15, help="# of blocks for the pixelated blurring method")
    args = vars(ap.parse_args())
    img = cv2.imread(args.image)
    start_time = time.time()
    output=face_blurring(img,ap.blur_type,ap.blocks)
    end_time = time.time()
    t = end_time-start_time
    print('time: {0}s'.format(t))
    cv2.imwrite(args.output, output)
