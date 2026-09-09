import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import mediapipe as mp
from streamlit_autorefresh import st_autorefresh  # 追加

from origami_tutor import STEPS, OrigamiTutor
import demo

st.set_page_config(page_title="折り紙チューター：ハート", layout="wide")

# ---------------------------------------------------------
# MediaPipe Hands の初期化（描画用）
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------------------------------------------------
# セッション状態の初期化
# ---------------------------------------------------------
if "tutor" not in st.session_state:
    st.session_state.tutor = OrigamiTutor(STEPS)

tutor = st.session_state.tutor

st.title("折り紙チューター：ハートの折り方")

# ---------------------------------------------------------
# 映像処理クラス
# ---------------------------------------------------------
class OrigamiProcessor(VideoProcessorBase):
    def __init__(self):
        self.step_num = 1
        self.is_ok = False
        
        # 描画用のMediaPipe Handsインスタンス
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1. demo.py による判定
        try:
            self.is_ok = demo.check_Origami(img, self.step_num)
        except Exception:
            self.is_ok = False

        # 2. 描画処理
        display = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 青色領域の検出
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([150, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel_size = (5, 5) if self.step_num in [1, 2] else (3, 3)
        kernel = np.ones(kernel_size, np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if blue_contours:
            largest_blue = max(blue_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_blue) > 1000:
                cv2.drawContours(display, [largest_blue], -1, (255, 0, 0), 2)

        # 外形ポリゴンの検出
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        outer_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in outer_contours:
            if cv2.contourArea(contour) < 10000:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            cv2.drawContours(display, [approx], -1, (0, 255, 0), 2)
            for pt in approx:
                cv2.circle(display, tuple(pt[0]), 4, (0, 255, 255), -1)
            
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(display, f"Vertices: {len(approx)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 手の検出
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 判定ステータスの描画
        color = (0, 255, 0) if self.is_ok else (0, 0, 255)
        text = f"Step {self.step_num}: {'OK' if self.is_ok else 'NG'}"
        cv2.putText(display, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

# 完了時の表示
if tutor.is_finished():
    st.balloons()
    st.success("🎉 おめでとうございます！ハートの折り紙が完成しました！")
    if st.button("最初からやり直す"):
        st.session_state.tutor = OrigamiTutor(STEPS)
        st.rerun()

else:
    # 500ミリ秒（0.5秒）ごとにメインスレッドの状態をミリ秒単位で確認
    st_autorefresh(interval=500, key="origami_step_checker")

    current_step_data = tutor.get_current_step()
    step_num = tutor.get_current_step_number()
    instruction = current_step_data["instruction"]
    total_steps = len(STEPS)

    st.subheader(f"Step {step_num} / {total_steps}")
    st.info(f"**指示:** {instruction}")

    col1, col2 = st.columns([1, 1])

    # ---------------------------------------------------------
    # 左カラム: カメラ入力
    # ---------------------------------------------------------
    with col1:
        ctx = webrtc_streamer(
            key="origami-cam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=OrigamiProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "aspectRatio": {"ideal": 16 / 9},
                },
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.step_num = step_num

            # OKフラグを検知したらステップを進めてフラグを下げる
            if ctx.video_processor.is_ok:
                ctx.video_processor.is_ok = False  # 連進を防ぐためフラグをリセット
                tutor.next_step()
                st.rerun()

    # ---------------------------------------------------------
    # 右カラム: 手動コントロール
    # ---------------------------------------------------------
    with col2:
        image_path = current_step_data.get("image")
        
        if image_path:
            st.image(image_path, caption=f"Step {step_num} のお手本", use_container_width=True)
        else:
            st.info("※ このステップの解説画像はありません。")

        st.divider()

        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("前のステップに戻る", use_container_width=True):
                if tutor.current_step > 0:
                    tutor.current_step -= 1
                    tutor.finished = False
                    st.rerun()

        with btn_col2:
            if st.button("強制的に次のステップへ", use_container_width=True):
                tutor.next_step()
                st.rerun()
