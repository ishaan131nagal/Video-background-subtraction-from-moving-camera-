import cv2

def background_subtraction(video_source='path/to/your/video.mp4'):
    # Step 1: Video Acquisition for Camera
    cap = cv2.VideoCapture(video_source)

    # Step 2: Background Subtractor Initialization
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    while True:
        # Step 3: Frame Acquisition
        ret, frame = cap.read()
        originalFrame = frame
        # Check if the video capture was successful
        if not ret:
            break

        # Step 4: Background Modeling and Subtraction
        foreground_mask = fgbg.apply(frame)

        # Step 5: Post-Processing (Morphological Operations)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Step 6: Object Detection (Contour Finding)
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 7: Drawing Bounding Boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Step 8: Bitwise AND operation between the frame and foreground mask
        result = cv2.bitwise_and(originalFrame, originalFrame, mask=foreground_mask)

        # Step 9: Display Results
        cv2.imshow('Original Frame', originalFrame)
        cv2.imshow('Foreground Mask', foreground_mask)
        cv2.imshow('Result', result)

        # Step 10: Break the loop if 'q' key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Step 11: Release Video Capture and Close Windows
    cap.release()
    cv2.destroyAllWindows()




# Specify the camera index
camera_index = 0  # Change the index accordingly

background_subtraction(camera_index)
