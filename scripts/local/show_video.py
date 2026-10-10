"""Load a video from a path and play it back in a window.

Usage:
    python scripts/local/show_video.py path/to/video.mp4

Press 'q' or Esc to quit early. Requires opencv-python and a display
(run locally, or over `ssh -X`/`ssh -Y` for X11 forwarding).
"""
import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", help="Path to the video file to play")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay_ms = max(1, round(1000 / fps))

    window_name = args.video_path
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):  # 'q' or Esc
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
