import cv2

file_path = './haarcascade_frontalface_default.xml'
classifier = cv2.CascadeClassifier(file_path)
misamo_path = 'misamo.jpg'
misamo_read = cv2.imread(misamo_path)

detect = classifier.detectMultiScale(
    misamo_read,
    minSize = (100,100),          
)

for x, y, w, h in detect:
    cv2.rectangle(
        misamo_read,
        (x,y),
        (x+w, y+h),     
        (0, 255, 0),   
        3,              
    )
    
cv2.imshow('Label Window', misamo_read)
cv2.waitKey(0)
cv2.destroyAllWindows()