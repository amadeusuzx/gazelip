import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
ret, frame = cap.read()
print(frame.shape)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
