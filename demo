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
        # 青色マスク
        # =========================================

        blue_mask = cv2.inRange(
            hsv,
            lower_blue,
            upper_blue
        )


        # =========================================
        # Stepごとの青色マスク処理
        # =========================================

        if i == 1 or i == 2:

            kernel = np.ones(
                (5, 5),
                np.uint8
            )

        else:

            kernel = np.ones(
                (3, 3),
                np.uint8
            )


        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # =========================================
        # 青色輪郭
        # =========================================

        contours, _ = cv2.findContours(
            blue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )


        # =========================================
        # 最大の青色領域
        # =========================================

        largest_contour = None

        if len(contours) > 0:

            contours = sorted(
                contours,
                key=cv2.contourArea,
                reverse=True
            )

            largest_contour = contours[0]

            blue_area = cv2.contourArea(
                largest_contour
            )


        # =====================================================
        # STEP 1
        # =====================================================

        if i == 1:

            # =========================================
            # 青色領域
            # =========================================

            if (
                largest_contour is not None
                and blue_area > 1000
            ):

                has_blue = True


                # =====================================
                # 青色三角形
                # =====================================

                perimeter = cv2.arcLength(
                    largest_contour,
                    True
                )

                approx = cv2.approxPolyDP(
                    largest_contour,
                    0.02 * perimeter,
                    True
                )

                if len(approx) == 3:

                    has_blue_triangle = True

                    cv2.drawContours(
                        display,
                        [approx],
                        -1,
                        (255, 0, 0),
                        3
                    )


            # =========================================
            # 外形検出
            # =========================================

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            edges = cv2.Canny(
                gray,
                50,
                150
            )

            edge_kernel = np.ones(
                (5, 5),
                np.uint8
            )

            edges = cv2.dilate(
                edges,
                edge_kernel
            )


            outer_contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


            for contour in outer_contours:

                area = cv2.contourArea(contour)

                if area < 10000:
                    continue

                perimeter = cv2.arcLength(
                    contour,
                    True
                )

                approx = cv2.approxPolyDP(
                    contour,
                    0.02 * perimeter,
                    True
                )

                if len(approx) == 5:

                    is_pentagon = True

                    cv2.drawContours(
                        display,
                        [approx],
                        -1,
                        (0, 255, 0),
                        3
                    )

                    break


        # =====================================================
        # STEP 2
        # =====================================================

        elif i == 2:

            # =========================================
            # 青色
            # =========================================

            if (
                largest_contour is not None
                and blue_area > 1000
            ):

                has_blue = True


                # =====================================
                # 青色の中心
                # =====================================

                M = cv2.moments(
                    largest_contour
                )

                if M["m00"] != 0:

                    blue_cx = (
                        M["m10"] /
                        M["m00"]
                    )

                    blue_cy = (
                        M["m01"] /
                        M["m00"]
                    )

                else:

                    blue_cx = 0
                    blue_cy = 0


            # =========================================
            # 外形検出
            # =========================================

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            edges = cv2.Canny(
                gray,
                50,
                150
            )

            edge_kernel = np.ones(
                (5, 5),
                np.uint8
            )

            edges = cv2.dilate(
                edges,
                edge_kernel
            )


            outer_contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


            hexagon = None


            for contour in outer_contours:

                area = cv2.contourArea(contour)

                if area < 10000:
                    continue


                # =====================================
                # 凸包
                # =====================================

                hull = cv2.convexHull(
                    contour
                )


                perimeter = cv2.arcLength(
                    hull,
                    True
                )

                approx = cv2.approxPolyDP(
                    hull,
                    0.03 * perimeter,
                    True
                )


                if len(approx) == 6:

                    is_hexagon = True

                    hexagon = approx

                    cv2.drawContours(
                        display,
                        [approx],
                        -1,
                        (0, 255, 0),
                        3
                    )

                    break


            # =========================================
            # 青色面積割合
            # =========================================

            if hexagon is not None:

                hexagon_area = cv2.contourArea(
                    hexagon
                )

                if hexagon_area > 0:

                    blue_ratio = (
                        blue_area /
                        hexagon_area
                    )


            blue_area_ok = (
                blue_ratio >= 0.50
            )


            # =========================================
            # 黄色マスク
            # =========================================

            yellow_mask = cv2.inRange(
                hsv,
                lower_yellow,
                upper_yellow
            )


            yellow_kernel = np.ones(
                (7, 7),
                np.uint8
            )

            yellow_mask = cv2.morphologyEx(
                yellow_mask,
                cv2.MORPH_OPEN,
                yellow_kernel
            )

            yellow_mask = cv2.morphologyEx(
                yellow_mask,
                cv2.MORPH_CLOSE,
                yellow_kernel
            )


            # =========================================
            # 黄色輪郭
            # =========================================

            yellow_contours, _ = cv2.findContours(
                yellow_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


            if hexagon is not None:

                for contour in yellow_contours:

                    area = cv2.contourArea(
                        contour
                    )

                    if area < 2000:
                        continue


                    perimeter = cv2.arcLength(
                        contour,
                        True
                    )

                    approx_yellow = cv2.approxPolyDP(
                        contour,
                        0.03 * perimeter,
                        True
                    )


                    if len(approx_yellow) != 4:
                        continue


                    M = cv2.moments(
                        contour
                    )

                    if M["m00"] == 0:
                        continue


                    cx = (
                        M["m10"] /
                        M["m00"]
                    )

                    cy = (
                        M["m01"] /
                        M["m00"]
                    )


                    inside = cv2.pointPolygonTest(
                        hexagon,
                        (float(cx), float(cy)),
                        False
                    )


                    if inside < 0:
                        continue


                    hex_M = cv2.moments(
                        hexagon
                    )

                    if hex_M["m00"] == 0:
                        continue


                    hex_center_x = (
                        hex_M["m10"] /
                        hex_M["m00"]
                    )


                    if cx < hex_center_x:

                        left_yellow = True

                    else:

                        right_yellow = True


                    cv2.drawContours(
                        display,
                        [approx_yellow],
                        -1,
                        (0, 255, 255),
                        3
                    )


            # =========================================
            # 黄色マスク表示
            # =========================================

            cv2.imshow(
                "Yellow Mask",
                yellow_mask
            )


        # =====================================================
        # STEP 3
        # =====================================================

        elif i == 3:

            if (
                largest_contour is not None
                and blue_area > 10000
            ):

                has_blue = True


                # =====================================
                # 六角形
                # =====================================

                perimeter = cv2.arcLength(
                    largest_contour,
                    True
                )

                approx = cv2.approxPolyDP(
                    largest_contour,
                    0.025 * perimeter,
                    True
                )


                if len(approx) == 6:

                    is_hexagon = True

                    points = approx.reshape(
                        -1,
                        2
                    )


                    cv2.drawContours(
                        display,
                        [approx],
                        -1,
                        (0, 255, 0),
                        3
                    )


                    # =================================
                    # 中心
                    # =================================

                    min_x = np.min(
                        points[:, 0]
                    )

                    max_x = np.max(
                        points[:, 0]
                    )

                    min_y = np.min(
                        points[:, 1]
                    )

                    max_y = np.max(
                        points[:, 1]
                    )

                    center_x = (
                        min_x + max_x
                    ) / 2

                    center_y = (
                        min_y + max_y
                    ) / 2


                    width = max_x - min_x


                    # =================================
                    # 一番下の頂点
                    # =================================

                    bottom_point = max(
                        points,
                        key=lambda p: p[1]
                    )

                    bottom_x = bottom_point[0]
                    bottom_y = bottom_point[1]


                    bottom_distance = abs(
                        bottom_x - center_x
                    )


                    bottom_threshold = (
                        width * 0.25
                    )


                    if bottom_distance < bottom_threshold:

                        has_bottom_point = True

                        cv2.circle(
                            display,
                            (
                                int(bottom_x),
                                int(bottom_y)
                            ),
                            14,
                            (0, 255, 255),
                            3
                        )


        # =====================================================
        # STEP 4
        # =====================================================

        elif i == 4:

            TARGET_VERTICES = 10
            BOTTOM_CENTER_THRESHOLD = 0.25


            if (
                largest_contour is not None
                and blue_area > 10000
            ):

                has_blue = True


                # =====================================
                # 周長
                # =====================================

                perimeter = cv2.arcLength(
                    largest_contour,
                    True
                )


                # =====================================
                # 10頂点になるepsilonを探す
                # =====================================

                approx = None

                for epsilon_ratio in np.arange(
                    0.001,
                    0.051,
                    0.001
                ):

                    epsilon = (
                        epsilon_ratio *
                        perimeter
                    )

                    temp_approx = cv2.approxPolyDP(
                        largest_contour,
                        epsilon,
                        True
                    )


                    if len(temp_approx) == TARGET_VERTICES:

                        approx = temp_approx

                        break


                # =====================================
                # 10頂点
                # =====================================

                if approx is not None:

                    is_heart_shape = True

                    vertex_count = len(
                        approx
                    )

                    points = approx.reshape(
                        -1,
                        2
                    )


                    # =================================
                    # 輪郭表示
                    # =================================

                    cv2.drawContours(
                        display,
                        [approx],
                        -1,
                        (0, 255, 0),
                        3
                    )


                    # =================================
                    # 頂点表示
                    # =================================

                    for n, point in enumerate(points):

                        x = int(point[0])
                        y = int(point[1])


                        cv2.circle(
                            display,
                            (x, y),
                            8,
                            (0, 0, 255),
                            -1
                        )


                        cv2.putText(
                            display,
                            str(n + 1),
                            (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )


                    # =================================
                    # 外形の中心
                    # =================================

                    min_x = np.min(
                        points[:, 0]
                    )

                    max_x = np.max(
                        points[:, 0]
                    )

                    center_x = (
                        min_x + max_x
                    ) / 2


                    # =================================
                    # 外形の幅
                    # =================================

                    width = max_x - min_x


                    # =================================
                    # 一番下の頂点
                    # =================================

                    bottom_point = max(
                        points,
                        key=lambda p: p[1]
                    )

                    bottom_x = bottom_point[0]
                    bottom_y = bottom_point[1]


                    # =================================
                    # 下の頂点と中心の距離
                    # =================================

                    bottom_distance = abs(
                        bottom_x - center_x
                    )


                    # =================================
                    # 下中央にあるか
                    # =================================

                    bottom_threshold = (
                        width *
                        BOTTOM_CENTER_THRESHOLD
                    )


                    if bottom_distance < bottom_threshold:

                        has_bottom_point = True

                        cv2.circle(
                            display,
                            (
                                int(bottom_x),
                                int(bottom_y)
                            ),
                            14,
                            (0, 255, 255),
                            3
                        )


        # =====================================================
        # MediaPipe Hands
        # =====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)


        if results.multi_hand_landmarks:

            hands_visible = True

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )


        # =====================================================
        # Stepごとの最終判定
        # =====================================================

        if i == 1:

            result = (
                has_blue
                and has_blue_triangle
                and is_pentagon
                and not hands_visible
            )


        elif i == 2:

            result = (
                has_blue
                and is_hexagon
                and blue_area_ok
                and left_yellow
                and right_yellow
                and not hands_visible
            )


        elif i == 3:

            result = (
                has_blue
                and is_hexagon
                and has_bottom_point
                and not hands_visible
            )


        elif i == 4:

            result = (
                has_blue
                and is_heart_shape
                and has_bottom_point
                and not hands_visible
            )


        else:

            print("iは1～4で指定してください")

            cap.release()
            hands.close()
            cv2.destroyAllWindows()

            return False


        # =====================================================
        # 100フレーム連続判定
        # =====================================================

        if result:

            true_count += 1

        else:

            true_count = 0


        # =====================================================
        # デバッグ表示
        # =====================================================

        cv2.putText(
            display,
            f"Step: {i}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"Blue: {has_blue}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        if i == 1:

            cv2.putText(
                display,
                f"Blue Triangle: {has_blue_triangle}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Pentagon: {is_pentagon}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        elif i == 2:

            cv2.putText(
                display,
                f"Hexagon: {is_hexagon}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Blue ratio: {blue_ratio:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Left Yellow: {left_yellow}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Right Yellow: {right_yellow}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        elif i == 3:

            cv2.putText(
                display,
                f"Hexagon: {is_hexagon}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Bottom point: {has_bottom_point}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        elif i == 4:

            cv2.putText(
                display,
                f"10 Vertices: {vertex_count == 10}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Bottom point: {has_bottom_point}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Bottom distance: {bottom_distance:.1f}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # =========================================
        # 手の表示
        # =========================================

        cv2.putText(
            display,
            f"Hand: {hands_visible}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # =========================================
        # True count
        # =========================================

        cv2.putText(
            display,
            f"True count: {true_count}/100",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # =========================================
        # 判定結果
        # =========================================

        if result:

            cv2.putText(
                display,
                "Checking...",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        else:

            cv2.putText(
                display,
                "FALSE",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )


        # =========================================
        # Blue Mask
        # =========================================

        cv2.imshow(
            "Blue Mask",
            blue_mask
        )


        # =========================================
        # カメラ映像
        # =========================================

        cv2.imshow(
            "Origami Detection",
            display
        )


        # =========================================
        # 100フレーム連続True
        # =========================================

        if true_count >= REQUIRED_TRUE_FRAMES:

            print()
            print("==============================")
            print(f"Step {i}: 正しい形です！")
            print("==============================")


            cv2.putText(
                display,
                "TRUE",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                4
            )


            cv2.imshow(
                "Origami Detection",
                display
            )

            cv2.waitKey(1000)

            cap.release()
            hands.close()
            cv2.destroyAllWindows()

            return True


        # =========================================
        # ESC
        # =========================================

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            break


    # =========================================
    # 終了処理
    # =========================================

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    return False


# =========================================
# 呼び出し用
# =========================================

def wait_for_cv_result(i):
    return check_Origami(i)


# =========================================
# テストする場合
# =========================================

if __name__ == "__main__":

    # 1～4のどれかを指定
    i = 4

    result = wait_for_cv_result(i)

    print("判定結果:", result)
