import time
import urllib.request
from pathlib import Path
import math

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyperclip


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path.home() / "AppData" / "Local" / "mpModels" / "hand_landmarker.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_MODEL_PATH = Path.home() / "AppData" / "Local" / "mpModels" / "face_landmarker.task"

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = Path.home() / "AppData" / "Local" / "mpModels" / "pose_landmarker_lite.task"

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def ensure_model(model_url, model_path, label):
    if not model_path.exists():
        print(f"{label} modeli indiriliyor...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        print(f"{label} modeli indirildi.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

KEY_ROWS = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["Ğ", "Ü", "Ş", "İ", "Ö", "Ç"],
    ["SPACE", "BKSP", "ENTER"],
]

TURKISH_KEY_OUTPUT = {
    "I": "ı",
    "Ğ": "ğ",
    "Ü": "ü",
    "Ş": "ş",
    "İ": "i",
    "Ö": "ö",
    "Ç": "ç",
}

KEYBOARD_INPUT_X_MIN = 0.25
KEYBOARD_INPUT_X_MAX = 0.75
KEYBOARD_INPUT_Y_MIN = 0.20
KEYBOARD_INPUT_Y_MAX = 0.80


def draw_keyboard(frame, finger_xy):
    """Tuşları çizer; parmağa en yakın tuşu vurgular ve döndürür."""
    height, width, _ = frame.shape

    top = int(height * 0.23)
    bottom = int(height * 0.96)
    row_gap = 6
    row_count = len(KEY_ROWS)
    row_height = (bottom - top - row_gap * (row_count - 1)) // row_count
    side_margin = int(width * 0.04)
    key_gap = 5

    key_rectangles = []

    for row_index, keys in enumerate(KEY_ROWS):
        y1 = top + row_index * (row_height + row_gap)
        y2 = y1 + row_height

        if row_index == len(KEY_ROWS) - 1:
            weights = [4, 2, 2]
        else:
            weights = [1] * len(keys)

        available_width = (
            width - 2 * side_margin - key_gap * (len(keys) - 1)
        )
        unit_width = available_width / sum(weights)
        row_width = int(
            unit_width * sum(weights) + key_gap * (len(keys) - 1)
        )
        x = (width - row_width) // 2

        for key, weight in zip(keys, weights):
            key_width = int(unit_width * weight)
            x1 = x
            x2 = x1 + key_width

            key_rectangles.append((key, x1, y1, x2, y2))
            x = x2 + key_gap

    hovered_key = None

    if finger_xy is not None:
        finger_x, finger_y = finger_xy
        hit_padding = 14
        best_distance = float("inf")

        for key, x1, y1, x2, y2 in key_rectangles:
            inside_expanded_area = (
                x1 - hit_padding <= finger_x <= x2 + hit_padding
                and y1 - hit_padding <= finger_y <= y2 + hit_padding
            )

            if inside_expanded_area:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                distance = (
                    (finger_x - center_x) ** 2
                    + (finger_y - center_y) ** 2
                )

                if distance < best_distance:
                    best_distance = distance
                    hovered_key = key

    for key, x1, y1, x2, y2 in key_rectangles:
        fill_color = (70, 130, 200) if key == hovered_key else (55, 55, 55)

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)

        font_scale = 0.65 if len(key) == 1 else 0.45
        text_size, _ = cv2.getTextSize(
            key,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2

        cv2.putText(
            frame,
            key,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
        )

    return hovered_key


def main():
    ensure_model(MODEL_URL, MODEL_PATH, "El takip")
    ensure_model(FACE_MODEL_URL, FACE_MODEL_PATH, "Yuz takibi")
    ensure_model(POSE_MODEL_URL, POSE_MODEL_PATH, "Vucut takibi")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    face_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print(
            "Kamera açılamadı. Kameranın başka bir uygulamada "
            "açık olmadığını kontrol et."
        )
        return

    pyautogui.FAILSAFE = True

    window_name = "MediaPipe - Sanal Klavye"
    screen_width, _ = pyautogui.size()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.moveWindow(window_name, max(0, screen_width - 660), 40)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    start_time = time.perf_counter()
    pinch_was_active = False
    last_key_time = 0.0
    pointer_x = None
    pointer_y = None
    pointer_smoothing = 0.25
    text_lines = [""]
    control_mode = "keyboard"  # Başlangıç modu
    mode_gesture_started = None
    mode_toggle_armed = True
    MODE_GESTURE_HOLD_SECONDS = 0.9
    last_click_time = 0.0

    with (
        mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker,
        mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        frame_count = 0
        face_result = None
        pose_result = None

        while True:
            success, frame = camera.read()

            if not success:
                print("Kameradan görüntü alınamadı.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Yuz ve vucut modellerini her karede degil, her uc karede calistir.
            # Boylece uc modelin ayni anda kamerayi yavaslatma olasiligi azalir.
            if frame_count % 3 == 0:
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_count += 1

            face_detected = bool(face_result and face_result.face_landmarks)
            pose_detected = bool(pose_result and pose_result.pose_landmarks)

            # Yuzun dis cevresini ve govde/kol/bacak iskeletini ciz.
            if face_detected:
                face = face_result.face_landmarks[0]
                face_points = [
                    (int(face[i].x * width), int(face[i].y * height))
                    for i in FACE_OVAL
                ]
                for i in range(len(face_points)):
                    cv2.line(
                        frame,
                        face_points[i],
                        face_points[(i + 1) % len(face_points)],
                        (255, 180, 0),
                        2,
                    )

            if pose_detected:
                pose = pose_result.pose_landmarks[0]
                for start_index, end_index in POSE_CONNECTIONS:
                    start_point = pose[start_index]
                    end_point = pose[end_index]
                    if (
                        getattr(start_point, "visibility", 1.0) > 0.4
                        and getattr(end_point, "visibility", 1.0) > 0.4
                    ):
                        cv2.line(
                            frame,
                            (int(start_point.x * width), int(start_point.y * height)),
                            (int(end_point.x * width), int(end_point.y * height)),
                            (255, 0, 255),
                            3,
                        )
                for point in pose:
                    if getattr(point, "visibility", 1.0) > 0.4:
                        cv2.circle(
                            frame,
                            (int(point.x * width), int(point.y * height)),
                            4,
                            (255, 255, 0),
                            -1,
                        )

            finger_xy = None
            is_pinching = False
            hand_state = "EL ALGILANMADI"
            extended_count = 0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                wrist = hand[0]

                finger_pairs = [
                    (8, 5),    # işaret parmağı
                    (12, 9),   # orta parmak
                    (16, 13),  # yüzük parmağı
                    (20, 17),  # serçe parmağı
                ]

                for tip_index, mcp_index in finger_pairs:
                    tip = hand[tip_index]
                    mcp = hand[mcp_index]

                    tip_distance = math.hypot(
                        tip.x - wrist.x,
                        tip.y - wrist.y,
                    )
                    base_distance = math.hypot(
                        mcp.x - wrist.x,
                        mcp.y - wrist.y,
                    )

                    if tip_distance > base_distance * 1.15:
                        extended_count += 1

                thumb_tip = hand[4]
                index_tip = hand[8]
                palm_middle = hand[9]

                pinch_distance = math.hypot(
                    thumb_tip.x - index_tip.x,
                    thumb_tip.y - index_tip.y,
                )
                palm_size = math.hypot(
                    palm_middle.x - wrist.x,
                    palm_middle.y - wrist.y,
                )

                pinch_ratio = pinch_distance / max(palm_size, 0.001)

                if pinch_was_active:
                    is_pinching = pinch_ratio < 0.30
                else:
                    is_pinching = pinch_ratio < 0.22

                if is_pinching:
                    hand_state = "SIKISTIRMA"
                elif extended_count >= 3:
                    hand_state = "ACIK EL"
                else:
                    hand_state = "KAPALI EL"

                # Yumruk 0.9 saniye tutulursa kontrol modu değişir.
                # Yeniden geçiş için önce elini açman gerekir.
                fist_gesture = extended_count == 0 and not is_pinching
                if fist_gesture:
                    if mode_gesture_started is None:
                        mode_gesture_started = time.perf_counter()
                    elif (
                        mode_toggle_armed
                        and time.perf_counter() - mode_gesture_started
                        >= MODE_GESTURE_HOLD_SECONDS
                    ):
                        control_mode = (
                            "cursor" if control_mode == "keyboard" else "keyboard"
                        )
                        mode_toggle_armed = False
                        pinch_was_active = False
                        mode_gesture_started = None
                else:
                    mode_gesture_started = None
                    if extended_count >= 3:
                        mode_toggle_armed = True

                mapped_x = (
                    index_tip.x - KEYBOARD_INPUT_X_MIN
                ) / (KEYBOARD_INPUT_X_MAX - KEYBOARD_INPUT_X_MIN)

                mapped_y = (
                    index_tip.y - KEYBOARD_INPUT_Y_MIN
                ) / (KEYBOARD_INPUT_Y_MAX - KEYBOARD_INPUT_Y_MIN)

                mapped_x = max(0.0, min(1.0, mapped_x))
                mapped_y = max(0.0, min(1.0, mapped_y))

                keyboard_top = int(height * 0.23)
                keyboard_bottom = int(height * 0.96)

                if pointer_x is None:
                    pointer_x = mapped_x
                    pointer_y = mapped_y
                else:
                    pointer_x += (mapped_x - pointer_x) * pointer_smoothing
                    pointer_y += (mapped_y - pointer_y) * pointer_smoothing

                keyboard_finger_xy = (
                    int(pointer_x * (width - 1)),
                    int(
                        keyboard_top
                        + pointer_y * (keyboard_bottom - keyboard_top)
                    ),
                )

                if control_mode == "keyboard":
                    finger_xy = keyboard_finger_xy
                else:
                    screen_width, screen_height = pyautogui.size()
                    cursor_screen_x = int(pointer_x * (screen_width - 1))
                    cursor_screen_y = int(pointer_y * (screen_height - 1))
                    pyautogui.moveTo(cursor_screen_x, cursor_screen_y)

                for start_index, end_index in HAND_CONNECTIONS:
                    start_point = hand[start_index]
                    end_point = hand[end_index]

                    start_xy = (
                        int(start_point.x * width),
                        int(start_point.y * height),
                    )
                    end_xy = (
                        int(end_point.x * width),
                        int(end_point.y * height),
                    )

                    cv2.line(frame, start_xy, end_xy, (0, 220, 0), 2)

                for landmark in hand:
                    point = (
                        int(landmark.x * width),
                        int(landmark.y * height),
                    )
                    cv2.circle(frame, point, 4, (0, 100, 255), -1)

                if control_mode == "keyboard":
                    cv2.circle(frame, finger_xy, 10, (255, 255, 0), -1)

            cv2.rectangle(
                frame,
                (0, 0),
                (width, int(height * 0.22)),
                (25, 25, 25),
                -1,
            )

            previous_line = text_lines[-2] if len(text_lines) > 1 else ""
            current_line = text_lines[-1]

            cv2.putText(
                frame,
                f"Yazi: {previous_line}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                frame,
                current_line[-45:],
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"El: {hand_state} | Yuz: {'VAR' if face_detected else 'YOK'} | Vucut: {'VAR' if pose_detected else 'YOK'}",
                (12, int(height * 0.19)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
            )

            if control_mode == "keyboard":
                hovered_key = draw_keyboard(frame, finger_xy)
                if finger_xy is not None:
                    cv2.circle(frame, finger_xy, 12, (0, 255, 255), 2)
            else:
                hovered_key = None

            # Önizleme imleç modunda üstte kalmasın; hedef uygulamaya tıklanabilsin.
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_TOPMOST,
                1 if control_mode == "keyboard" else 0,
            )

            if is_pinching and not pinch_was_active:
                now = time.perf_counter()

                if control_mode == "cursor":
                    if now - last_click_time > 0.35:
                        pyautogui.click()
                        last_click_time = now
                elif hovered_key and now - last_key_time > 0.25:
                    if hovered_key == "SPACE":
                        text_lines[-1] += " "
                        pyautogui.press("space")

                    elif hovered_key == "BKSP":
                        text_lines[-1] = text_lines[-1][:-1]
                        pyautogui.press("backspace")

                    elif hovered_key == "ENTER":
                        text_lines.append("")
                        pyautogui.press("enter")

                    else:
                        character = TURKISH_KEY_OUTPUT.get(
                            hovered_key,
                            hovered_key.lower(),
                        )
                        text_lines[-1] += character

                        if character.isascii():
                            pyautogui.write(character)
                        else:
                            pyperclip.copy(character)
                            pyautogui.hotkey("ctrl", "v")

                    last_key_time = now

            pinch_was_active = is_pinching

            mode_label = "IMLEC" if control_mode == "cursor" else "YAZI"
            if mode_gesture_started is not None:
                gesture_hint = " | Yumrugu tut: mod degistir"
            else:
                gesture_hint = " | Mod icin yumrugu 0.9 sn tut"

            cv2.putText(
                frame,
                f"Mod: {mode_label} | Yumruk: mod degistir | Cikmak icin Q",
                (12, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

