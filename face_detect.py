import cv2


def calculate_dim(frame,w_percent,h_percent):
        height = int(frame.shape[1] * (h_percent) / 100)
        width = int(frame.shape[0] * (w_percent) / 100)
        return (width, height)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("Experiment.mp4")

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        dim = calculate_dim(gray,50,20)
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

        faces = faceCascade.detectMultiScale(resized,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
        print("Found {0} faces!".format(len(faces)))
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture",resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:break
cap.release()
cv2.destroyAllWindows()

