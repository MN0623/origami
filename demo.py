import cv2
import numpy as np
import mediapipe as mp


# =========================================
# MediaPipe Hands
# =========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# =========================================
# Stepごとの判定
# =========================================

def check_Origami(i):

    # =========================================
    # カメラ
    # =========================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けませんでした")
        return False


    # =========================================
    # MediaPipe Hands
    # =========================================

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


    # =========================================
    # 共通設定
    # =========================================

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])

    REQUIRED_TRUE_FRAMES = 100


    # =========================================
    # Step2用
    # =========================================

    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([40, 255, 255])


    # =========================================
    # 連続Trueカウント
    # =========================================

    true_count = 0


    # =========================================
    # メインループ
    # =========================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # =========================================
        # 左右反転
        # =========================================

        frame = cv2.flip(frame, 1)

        display = frame.copy()


        # =========================================
        # HSV変換
        # =========================================

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        # =========================================
        # 共通初期値
        # =========================================

        has_blue = False
        has_blue_triangle = False

        is_pentagon = False
        is_hexagon = False
        is_heart_shape = False

        has_bottom_point = False

        blue_area = 0
        blue_ratio = 0

        vertex_count = 0
        bottom_distance = 0

        left_yellow = False
        right_yellow = False

        hands_visible = False


        # =========================================
        # 青色マスクimport cv2
import numpy as np
import mediapipe as mp


# =========================================
# MediaPipe Hands
# =========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================
# 折り紙判定
# =========================================

def check_Origami(img, i):

    # =====================================
    # 手の検出
    # =====================================

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    hands_visible = (
        results.multi_hand_landmarks is not None
    )

    no_hands = not hands_visible


    # =====================================
    # グレースケール
    # =====================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================
    # GaussianBlur
    # 細かい凹凸・ノイズを滑らかにする
    # =====================================

    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # =====================================
    # Canny
    # =====================================

    edges = cv2.Canny(
        gray_blur,
        50,
        150
    )


    # =====================================
    # MORPH_CLOSE
    # 輪郭の切れ目をつなげる
    # =====================================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )


    # =====================================
    # 輪郭検出
    # =====================================

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # =====================================
    # 小さいノイズを除去
    # =====================================

    contours = [
        cnt
        for cnt in contours
        if cv2.contourArea(cnt) >= 10000
    ]


    # =====================================
    # 初期値
    # =====================================

    shape_found = False
    bottom_center = False

    vertex_count = 0


    # =====================================
    # Step 1
    # 五角形
    # =====================================

    if i == 1:

        for cnt in contours:

            perimeter = cv2.arcLength(
                cnt,
                True
            )

            approx = cv2.approxPolyDP(
                cnt,
                0.02 * perimeter,
                True
            )

            vertices = len(approx)

            if vertices == 5:

                shape_found = True
                vertex_count = vertices

                break


        result = (
            shape_found
            and no_hands
        )


    # =====================================
    # Step 2
    # 六角形
    # =====================================

    elif i == 2:

        for cnt in contours:

            perimeter = cv2.arcLength(
                cnt,
                True
            )

            approx = cv2.approxPolyDP(
                cnt,
                0.03 * perimeter,
                True
            )

            vertices = len(approx)

            if vertices == 6:

                shape_found = True
                vertex_count = vertices

                break


        result = (
            shape_found
            and no_hands
        )


    # =====================================
    # Step 3
    # 六角形
    # ＋ 下側の頂点が中央
    # =====================================

    elif i == 3:

        for cnt in contours:

            perimeter = cv2.arcLength(
                cnt,
                True
            )

            approx = cv2.approxPolyDP(
                cnt,
                0.025 * perimeter,
                True
            )

            vertices = len(approx)

            if vertices != 6:
                continue


            shape_found = True
            vertex_count = vertices


            # -----------------------------
            # 頂点座標
            # -----------------------------

            points = approx.reshape(-1, 2)

            xs = points[:, 0]
            ys = points[:, 1]


            # -----------------------------
            # 一番下の頂点
            # -----------------------------

            bottom_index = np.argmax(ys)

            bottom_x = xs[bottom_index]


            # -----------------------------
            # 中央
            # -----------------------------

            min_x = np.min(xs)
            max_x = np.max(xs)

            center_x = (
                min_x + max_x
            ) / 2

            width = max_x - min_x


            # -----------------------------
            # 下側頂点が中央付近か
            # -----------------------------

            distance = abs(
                bottom_x - center_x
            )

            bottom_center = (
                distance < width * 0.25
            )

            break


        result = (
            shape_found
            and bottom_center
            and no_hands
        )


    # =====================================
    # Step 4
    # 10頂点
    # ＋ 下側の頂点が中央
    # =====================================

    elif i == 4:

        for cnt in contours:

            perimeter = cv2.arcLength(
                cnt,
                True
            )

            found_10 = None


            # -----------------------------
            # 10頂点になる近似値を探す
            # -----------------------------

            for epsilon_ratio in np.arange(
                0.001,
                0.051,
                0.001
            ):

                approx = cv2.approxPolyDP(
                    cnt,
                    epsilon_ratio * perimeter,
                    True
                )

                if len(approx) == 10:

                    found_10 = approx
                    break


            if found_10 is None:
                continue


            approx = found_10

            shape_found = True
            vertex_count = 10


            # -----------------------------
            # 頂点座標
            # -----------------------------

            points = approx.reshape(-1, 2)

            xs = points[:, 0]
            ys = points[:, 1]


            # -----------------------------
            # 一番下の頂点
            # -----------------------------

            bottom_index = np.argmax(ys)

            bottom_x = xs[bottom_index]


            # -----------------------------
            # 中央
            # -----------------------------

            min_x = np.min(xs)
            max_x = np.max(xs)

            center_x = (
                min_x + max_x
            ) / 2

            width = max_x - min_x


            # -----------------------------
            # 下側頂点が中央付近
            # -----------------------------

            distance = abs(
                bottom_x - center_x
            )

            bottom_center = (
                distance < width * 0.25
            )

            break


        result = (
            shape_found
            and bottom_center
            and no_hands
        )


    # =====================================
    # Step番号エラー
    # =====================================

    else:

        raise ValueError(
            "Step番号は1～4で指定してください。"
        )


    # =====================================
    # 判定結果を返す
    # =====================================

    return result
