import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()


    # cv2.rectangle(frame, (100,100), (300,300), (0, 0, 255), 3)
    # cv2.line(frame, (0,0), (400, 400), (255, 0, 0), 2)
    # cv2.circle(frame, (50, 50), 10, (0, 255, 0), -1)
    cv2.putText(frame, 'Talha', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()