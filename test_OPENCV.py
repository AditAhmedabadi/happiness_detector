import cv2
print(cv2.__version__)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')     #cascade for detecting face

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')                      #cascade for detecting eyes

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')                  #cascade for detecting smile

def detect(gray , frame):
    faces = face_cascade.detectMultiScale(gray , scaleFactor= 1.3 , minNeighbors= 5)    #face coordinates - tuple
    for (x , y  , w , h) in faces:
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255 , 0 , 0) , thickness= 2)      #creating rectangle around the face when it is detected
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x : x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray , scaleFactor= 1.1 , minNeighbors= 22)  
        for (ex , ey , ew , eh) in eyes:
            cv2.rectangle(roi_color , (ex,  ey) , (ex+ew , ey+eh) , (0,255,0) , 2)
        smile = smile_cascade.detectMultiScale(roi_gray , scaleFactor= 1.9 , minNeighbors=25)
        for (sx , sy , sw , sh) in smile:
            cv2.rectangle(roi_color , (sx , sy) ,(sx + sw , sy+sh) , (0,0,255) , thickness=2)
            if sx:
                cv2.putText(frame  , "EMOTION : HAPPY" , (50, 50 ) , fontFace = cv2.FONT_HERSHEY_SIMPLEX  ,fontScale = 2 , color = ( 0 ,0 ,0 )  , thickness= 2 )

    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _ , frame = video_capture.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    canvas = detect(gray , frame)
    cv2.imshow('Video' , canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()