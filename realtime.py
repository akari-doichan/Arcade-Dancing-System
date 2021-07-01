#coding:utf-8
import cv2
import argparse
import mediapipe as mp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="realtime-demo", add_help=True)
    parser.add_argument("--cam",         type=int, default=0, help="Camera number")
    parser.add_argument("--window-name", type=str, default="Realtime Demo", help="Window Name")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    winname = args.window_name

    print("Preparing mediapipe...")    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        while True:
            is_ok, frame = cap.read()
            if ((not is_ok) or (frame is None)):
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    # landmark_drawing_spec=mp_drawing.DrawingSpec(color=(236,163,245), thickness=10, circle_radius=10),
                    # connection_drawing_spec=mp_drawing.DrawingSpec(color=(236,163,245), thickness=10, circle_radius=2),
                )
            cv2.imshow(winname, frame)
            key = cv2.waitKey(1)
            if (key == 27) or (key==ord("q")):
                break
    cap.release()
    cv2.destroyWindow(winname)