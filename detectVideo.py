
import cv2
from ultralytics import YOLO
import time

def detectVideo():
    # Load the SNN model
    model = YOLO("best2.pt")

    # Open the camera/video (drone videos are saved in the testVideos/ folder)
    cap = cv2.VideoCapture('testVideos/720CityVideo.mp4')
    # cap = cv2.VideoCapture(0)

    # Average time
    timeSum = 0
    frameCounter = 0

    # Loop through the video frames
    while cap.isOpened():
        # start timer
        timer = time.time()

        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run SNN detection on the frame
            results = model.track(frame, persist=True) # Track (better for videos?)
            # results = model(frame) # Inference (For images)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Time per frame
            currTime = round((time.time() - timer) * 1000, 2)
            print('Time for frame: {}'.format(currTime))
            timeSum += currTime
            frameCounter += 1

            # Display the annotated frame
            cv2.imshow("SNN Detection", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print('Average time: {}'.format(timeSum / frameCounter))

if __name__ == '__main__':
    detectVideo()
