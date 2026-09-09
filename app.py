import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import mediapipe as mp
from streamlit_autorefresh import st_autorefresh  # 追加

from origami_tutor import STEPS, OrigamiTutor
import demo

st.set_page_config(page_title="Origami tutor：Heart", layout="wide")

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

st.title("Origami tutor：how to make a heart")

# ---------------------------------------------------------
# 映像処理クラス
# ---------------------------------------------------------
# ---------------------------------------------------------
# 映像処理クラス
# ---------------------------------------------------------
class OrigamiProcessor(VideoProcessorBase):
    def __init__(self):
        self.step_num = 1
        self.is_ok = False
        
        # --- 【追加】連続判定用カウンター ---
        self.ok_counter = 0
        self.REQUIRED_FRAMES = 8  # 8フレーム連続OKで達成（約0.25〜0.3秒間キープ）
        
        # 描画用のMediaPipe Handsインスタンス
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def reset_counter(self):
        """ステップ遷移時などにカウンターをリセットするメソッド"""
        self.ok_counter = 0
        self.is_ok = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1. demo.py による単一フレーム判定
        try:
            current_frame_ok = demo.check_Origami(img, self.step_num)
        except Exception:
            current_frame_ok = False

        # --- 【追加】連続成功カウント処理 ---
        if current_frame_ok:
            self.ok_counter += 1
        else:
            self.ok_counter = 0  # 判定が切れたら即リセット

        # 規定のフレーム数を超えて保持された場合のみ OK フラグを立てる
        self.is_ok = (self.ok_counter >= self.REQUIRED_FRAMES)

        # 2. 描画処理 (可視化用テキストに連続フレーム数を出すと分かりやすい)
        display = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

        # --- 判定ステータスの描画（進行カウントプログレスを表示） ---
        color = (0, 255, 0) if self.is_ok else (0, 0, 255)
        text = f"Step {self.step_num}: {'OK' if self.is_ok else 'NG'} ({min(self.ok_counter, self.REQUIRED_FRAMES)}/{self.REQUIRED_FRAMES})"
        cv2.putText(display, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

# 完了時の表示
step_num = tutor.get_current_step_number()

# Step 5 に到達した場合（完成）
if step_num == 5:
    st.balloons()  # 紙吹雪・風船演出
    st.success("🎉 finished！")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Step 5 のお手本画像を表示
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
        st.write("Did you enjoy it？")
        st.divider()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Back(STEP4)", use_container_width=True):
                tutor.current_step = 3  # Step 4 (インデックス 3) に戻す
                tutor.finished = False
                st.rerun()

        with btn_col2:
            if st.button("Restart ", use_container_width=True):
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
    st.info(f"**instructions : ** {instruction}")

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
            # 規定フレーム数連続でOKフラグが立った場合
            if ctx.video_processor.is_ok:
                ctx.video_processor.reset_counter()  # カウンターとOKフラグを初期化
                tutor.next_step()
                st.rerun()

    # ---------------------------------------------------------
    # 右カラム: 手動コントロール
    # ---------------------------------------------------------
    with col2:
        image_path = current_step_data.get("image")
        
        if image_path:
            st.image(image_path, caption=f"Example for Step {step_num} ", use_container_width=True)
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
