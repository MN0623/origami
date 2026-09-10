import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import demo
from origami_tutor import STEPS, OrigamiTutor

st.set_page_config(page_title="Origami tutor：Heart", layout="wide")

# MediaPipe Hands の初期化
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# セッション状態の初期化
if "tutor" not in st.session_state:
    st.session_state.tutor = OrigamiTutor(STEPS)

tutor = st.session_state.tutor

st.title("Origami tutor：how to make a heart")


# ---------------------------------------------------------
# 映像処理クラス (軽量化・メモリ最適化版)
# ---------------------------------------------------------
class OrigamiProcessor(VideoProcessorBase):

    def __init__(self):
        self.step_num = 1
        self.is_ok = False
        self.ok_counter = 0
        self.REQUIRED_FRAMES = 8
        self.frame_count = 0

        # --- 【追加】前回の検出結果を保持する変数 ---
        self.last_approx_list = []
        self.last_hand_landmarks = None

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset_counter(self):
        self.ok_counter = 0
        self.is_ok = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        # 1. 判定処理 (demo.py)
        try:
            current_frame_ok = demo.check_Origami(img, self.step_num)
        except Exception:
            current_frame_ok = False

        if current_frame_ok:
            self.ok_counter += 1
        else:
            self.ok_counter = 0

        self.is_ok = self.ok_counter >= self.REQUIRED_FRAMES

        display = img.copy()

        # ---------------------------------------------------------
        # 重い「検出処理」は 2フレームに1回だけ実行して結果を保存
        # ---------------------------------------------------------
        if self.frame_count % 2 == 0:
            # --- 輪郭の計算 ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
            outer_contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            self.last_approx_list = []
            for contour in outer_contours:
                if cv2.contourArea(contour) < 10000:
                    continue
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                self.last_approx_list.append(approx)

            # --- 手の計算 ---
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                self.last_hand_landmarks = results.multi_hand_landmarks
            else:
                self.last_hand_landmarks = None

        # ---------------------------------------------------------
        # 「描画処理」は前回の結果を使って【毎フレーム】実行する
        # （これでチカチカしなくなります！）
        # ---------------------------------------------------------
        # 1. 輪郭の描画
        for approx in self.last_approx_list:
            cv2.drawContours(display, [approx], -1, (0, 255, 0), 2)
            for pt in approx:
                cv2.circle(display, tuple(pt[0]), 4, (0, 255, 255), -1)

            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(
                display,
                f"Vertices: {len(approx)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 3. ステータス描画
        color = (0, 255, 0) if self.is_ok else (0, 0, 255)
        text = f"Step {self.step_num}: {'OK' if self.is_ok else 'NG'} ({min(self.ok_counter, self.REQUIRED_FRAMES)}/{self.REQUIRED_FRAMES})"
        cv2.putText(
            display,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        return av.VideoFrame.from_ndarray(display, format="bgr24")

# ---------------------------------------------------------
# メイン画面処理
# ---------------------------------------------------------
step_num = tutor.get_current_step_number()

if step_num == 5:
    st.balloons()
    st.success("🎉 finished！")

    col1, col2 = st.columns([1, 1])
    with col1:
        current_step_data = tutor.get_current_step()
        image_path = current_step_data.get("image")
        if image_path:
            st.image(
                image_path,
                caption="🎉 example (Step 5)",
                use_container_width=True,
            )
        else:
            st.info("No image")

    with col2:
        st.write("### Good job！")
        st.divider()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Back(STEP4)", use_container_width=True):
                tutor.current_step = 3
                tutor.finished = False
                st.rerun()

        with btn_col2:
            if st.button("Restart ", use_container_width=True):
                st.session_state.tutor = OrigamiTutor(STEPS)
                st.rerun()
else:
    # リフレッシュ間隔を1000ms (1秒) に広げて全体再描画の負荷を軽減
    st_autorefresh(interval=1000, key="origami_step_checker")

    current_step_data = tutor.get_current_step()
    instruction = current_step_data["instruction"]
    total_steps = len(STEPS)

    st.subheader(f"Step {step_num} / {total_steps}")
    st.info(f"**instructions : ** {instruction}")
    st.write("Once finished, move your hands out of the frame to show the full origami.")

    col1, col2 = st.columns([1, 1])

    with col1:
        ctx = webrtc_streamer(
            key="origami-cam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=OrigamiProcessor,
            media_stream_constraints={
                "video": {
                    # 解像度を 640x480 に落としてメモリと通信量を大幅削減
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15},  # FPSを抑えるのも効果的
                },
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.step_num = step_num
            if ctx.video_processor.is_ok:
                ctx.video_processor.reset_counter()
                tutor.next_step()
                st.rerun()

    with col2:
        image_path = current_step_data.get("image")
        if image_path:
            st.image(
                image_path,
                caption=f"Example for Step {step_num} ",
                use_container_width=True,
            )
        else:
            st.info("No image")

        st.divider()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Back", use_container_width=True):
                if tutor.current_step > 0:
                    tutor.current_step -= 1
                    tutor.finished = False
                    st.rerun()

        with btn_col2:
            if st.button("Next step", use_container_width=True):
                tutor.next_step()
                st.rerun()
