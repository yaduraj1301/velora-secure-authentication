import cv2

# Initialize video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Unable to access the camera.")
else:
    print("Camera is working!")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Show the captured frame
        cv2.imshow('Video', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
