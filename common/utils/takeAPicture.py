import face_recognition
import cv2

if __name__ == '__main__':
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("rtsp://192.168.0.219:554/media1.sdp")

    counter = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_locations = face_recognition.face_locations(gray, number_of_times_to_upsample=1, model="cnn")

        frame_cut = frame

        ''' Check if only one face is detected '''
        if len(face_locations) == 1:
            top, right, bottom, left = face_locations[0]
            frame_cut=frame[top:bottom,left:right]
            #cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)

        cv2.imshow("frame", frame)
        cv2.imshow("frame_cut", frame_cut)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            frame_cut = cv2.resize(frame_cut, (220, 220))
            cv2.imwrite("../faces/" + str(counter) + ".jpg", frame_cut)
            counter += 1
    cap.release()
    cv2.destroyAllWindows()
