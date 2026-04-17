import cv2
import mediapipe as mp
import vlc
import time
import numpy as np
import os

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Configuration ---
VIDEO_PATH_DIR = "./videos"
VIDEO_FILE_EXTENSIONS = (".mp4", ".mkv")
MODEL_PATH = "./face_landmarker.task"  # Model originally provided by Google/MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models
GAZE_THRESHOLD_YAW = 40  # Degrees head can turn before it's "looking away"
GAZE_THRESHOLD_PITCH = 15  # Degrees head can tilt before it's "looking away"
COOLDOWN_SECONDS = 0.5  # Buffer to prevent flickering play/pause

DEBUG_MODE = False  # Set to True to see the head pose axes drawn on the video feed
DEBUG_MS_INTERVAL = 500  # Minimum milliseconds between showing debug images. Minimizes performance impact, especially when on RPi
last_debug_image_timestamp = 0
debug_image_queue = None  # Global to hold the latest debug image

instance = vlc.Instance()
media_list = instance.media_list_new()  # type: ignore
list_player = instance.media_list_player_new()  # type: ignore

last_look_time = time.time()
is_paused = False


def get_video_source():
    print("[*] Trying Pi Camera first...\n")

    try:
        from picamera2 import Picamera2  # type: ignore

        picam2 = Picamera2()
        picam2.configure(
            picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
        )
        picam2.start()
        return picam2, True
    except ImportError:
        print("Error setting up pi camera")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("\n[*] USB Webcam detected.\n")
        return cap, False

    cap.release()
    raise RuntimeError("No video source found.")


def print_result(result, output_image: mp.Image, timestamp_ms: int):
    global is_paused, last_look_time, debug_image_queue

    img_h, img_w, _ = output_image.numpy_view().shape

    looking_at_screen = False

    for _, face_blendshapes in enumerate(result.face_blendshapes):
        if face_blendshapes:
            # Blendshapes are a list of categories
            blendshapes = {b.category_name: b.score for b in face_blendshapes}

            # Threshold for "Looking Away" (typically 0.4 to 0.6)
            threshold = 0.5

            look_left = (
                blendshapes["eyeLookOutLeft"] > threshold
                or blendshapes["eyeLookInRight"] > threshold
            )
            look_right = (
                blendshapes["eyeLookInLeft"] > threshold
                or blendshapes["eyeLookOutRight"] > threshold
            )
            look_up = (
                blendshapes["eyeLookUpLeft"] > threshold
                or blendshapes["eyeLookUpRight"] > threshold
            )
            look_down = (
                blendshapes["eyeLookDownLeft"] > threshold
                or blendshapes["eyeLookDownRight"] > threshold
            )

            if not (look_left or look_right or look_up or look_down):
                looking_at_screen = True

    if not looking_at_screen or DEBUG_MODE:
        if DEBUG_MODE and not looking_at_screen:
            print("[*] No strong eye gaze detected, checking head pose as fallback...")
        for i, face_landmarks in enumerate(result.face_landmarks):
            if face_landmarks:
                debug_image = (
                    np.copy(output_image.numpy_view())
                    if DEBUG_MODE
                    and timestamp_ms - last_debug_image_timestamp > DEBUG_MS_INTERVAL
                    else None
                )
                pitch, yaw = get_head_pose(face_landmarks, img_w, img_h, debug_image)

                # in debug mode, we only debug the first detected face to avoid clutter
                if DEBUG_MODE and i == 0 and debug_image is not None:
                    debug_image_queue = debug_image  # Store for main thread

                    print(f"[*] Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")

                # Logic: Is the head pointed roughly at the camera/screen?
                if abs(yaw) < GAZE_THRESHOLD_YAW or abs(pitch) < GAZE_THRESHOLD_PITCH:
                    looking_at_screen = True

    # State Control
    if looking_at_screen:
        if not is_paused:
            list_player.pause()
            is_paused = True
        last_look_time = time.time()
    else:
        # Only resume if nobody has looked for the duration of the cooldown
        if is_paused and (time.time() - last_look_time > COOLDOWN_SECONDS):
            list_player.play()
            is_paused = False


def get_head_pose(landmarks, img_w, img_h, debug_image=None):
    """
    Estimates head orientation and, optionally, draws 3D axes for debugging.

    High Level: This function translates 2D screen coordinates into 3D spatial orientation.

    Low Level: Uses the cv2.solvePnP algorithm to find the rotation vector of a 3D model
    that best fits the observed 2D landmark points.
    """

    face_landmarks = landmarks

    # --- STEP 1: Define the Reference 3D Object ---
    # These coordinates represent a standard human face in 3D space (X, Y, Z).
    # The nose tip (0,0,0) is our anchor point for the entire coordinate system.
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin (down and slightly back)
            (
                -225.0,
                170.0,
                -135.0,
            ),  # Left eye left corner (left, up, and further back)
            (
                225.0,
                170.0,
                -135.0,
            ),  # Right eye right corner (right, up, and further back)
            (-150.0, -150.0, -125.0),  # Left Mouth corner (left, down, and back)
            (150.0, -150.0, -125.0),  # Right mouth corner (right, down, and back)
        ]
    )

    # --- STEP 2: Extract Corresponding 2D Points ---
    # We map the specific MediaPipe landmark indices to our 3D model points above.
    indices = [1, 152, 33, 263, 61, 291]

    # We convert 'normalized' coordinates (0.0 to 1.0) into actual pixel values.
    # face_landmarks[i].x * img_w gives the horizontal pixel position.
    image_points = np.array(
        [(face_landmarks[i].x * img_w, face_landmarks[i].y * img_h) for i in indices],
        dtype="double",
    )

    # --- STEP 3: Estimate Camera Properties ---
    # To do 3D math, we need to know how the camera "sees."
    # Since we don't have the exact lens specs, we approximate the 'Camera Matrix'.
    focal_length = img_w  # Assumption: focal length is roughly the width of the image.
    center = (
        img_w / 2,
        img_h / 2,
    )  # The optical center is usually the middle of the frame.

    # This matrix describes the internal parameters of the camera (intrinsic matrix).
    # It follows the standard pinhole camera model:
    # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    # We assume 'zero' lens distortion (dist_coeffs) for simplicity on a webcam/Pi cam.
    dist_coeffs = np.zeros((4, 1))

    # --- STEP 4: Solve the PnP Problem ---
    # This is the heavy lifting. It finds the rotation (rvec) and translation (tvec)
    # that projects our 3D 'model_points' onto our 2D 'image_points'.
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    # --- 5. Project 3D Axes onto the 2D Image ---
    # We define 3 lines starting from the nose (0,0,0):
    # X (blue), Y (green), and Z (red - the "look" vector)
    axis_length = 300.0
    axis_points_3d = np.array(
        [
            (axis_length, 0, 0),  # X axis point
            (0, axis_length, 0),  # Y axis point
            (0, 0, axis_length),  # Z axis point (Pointing OUT of the nose)
        ]
    )

    # Project these 3D points into 2D pixel coordinates
    (projected_points, _) = cv2.projectPoints(
        axis_points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )

    # Drawing Logic (if an image was provided)
    if debug_image is not None:
        p1 = (int(image_points[0][0]), int(image_points[0][1]))  # Nose tip pixel

        # Draw X (Blue), Y (Green), Z (Red)
        cv2.line(
            debug_image,
            p1,
            (int(projected_points[0][0][0]), int(projected_points[0][0][1])),
            (255, 0, 0),
            2,
        )
        cv2.line(
            debug_image,
            p1,
            (int(projected_points[1][0][0]), int(projected_points[1][0][1])),
            (0, 255, 0),
            2,
        )
        cv2.line(
            debug_image,
            p1,
            (int(projected_points[2][0][0]), int(projected_points[2][0][1])),
            (0, 0, 255),
            3,
        )

    # --- 6. Convert Vectors to Readable Angles ---
    # The rotation_vector is in a compact format (Rodrigues). We expand it to a 3x3 matrix.
    rmat, _ = cv2.Rodrigues(rotation_vector)

    # We decompose that matrix into three Euler angles (Pitch, Yaw, Roll).
    # RQDecomp3x3 breaks down the combined rotation into its individual axis components.
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    # Result:
    # angles[0] = Pitch (Nodding up/down)
    # angles[1] = Yaw (Shaking head left/right)
    # angles[2] = Roll (Tilting head like a confused dog)
    return angles[0], angles[1]


if __name__ == "__main__":
    cap_obj, is_pi_cam = get_video_source()

    # Initialize MediaPipe Face Mesh
    # Create a face landmarker instance with the live stream mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        result_callback=print_result,
        min_face_detection_confidence=0.3,  # Lower this for faster initial lock
        min_face_presence_confidence=0.3,  # Lower this for faster tracking
        min_tracking_confidence=0.3,  # How hard it "clings" to the face
    )
    face_detector = FaceLandmarker.create_from_options(options)

    # 3. Add files from directory to the media list
    for file_name in os.listdir(VIDEO_PATH_DIR):
        if file_name.endswith(VIDEO_FILE_EXTENSIONS):
            file_path = os.path.join(VIDEO_PATH_DIR, file_name)
            media = instance.media_new(file_path)  # type: ignore
            media_list.add_media(media)

    # 4. Set the list and play
    list_player.set_media_list(media_list)
    list_player.get_media_player().set_fullscreen(True)

    try:
        list_player.play()

        if not is_pi_cam and not cap_obj.isOpened():
            print("\nError: Could not open webcam.")

        while True:
            if is_pi_cam:
                image = cap_obj.capture_array()  # type: ignore
                success = True
            else:
                success, image = cap_obj.read()

            if not success:
                break

            # Convert the frame received from OpenCV/PiCamera2 to a MediaPipe’s Image object.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            face_detector.detect_async(mp_image, timestamp_ms)

            if (
                DEBUG_MODE
                and debug_image_queue is not None
                and timestamp_ms - last_debug_image_timestamp > DEBUG_MS_INTERVAL
            ):  # only show debug image if at least one second has passed since last debug image shown
                cv2.imshow("Debug View", debug_image_queue)
                last_debug_image_timestamp = timestamp_ms
                debug_image_queue = None  # Clear after showing

            # Press 'q' to exit, focus must be on the OpenCV GUI window
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        if is_pi_cam:
            cap_obj.stop()  # type: ignore
        else:
            cap_obj.release()
        list_player.stop()
