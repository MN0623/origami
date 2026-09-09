import time
#from check_origami import check_origami


# =========================================================
# 折り紙チューター：ハートの折り方
# 手順管理プログラム
# =========================================================


# ---------------------------------------------------------
# ① 折り紙の手順
# ---------------------------------------------------------

STEPS = [
    {
        "step": 1,
        "instruction": "Fold the top corner down to the center crease.",
        "image": "images/step1.png"
    },

    {
        "step": 2,
        "instruction": "Fold the bottom corner up to the crease on the top edge.",
        "image": "images/step2.png"
    },

    {
        "step": 3,
        "instruction": "Fold the lower left and lower right edges diagonally toward the center crease.",
        "image": "images/step3.png"
    },

    {
        "step": 4,
        "instruction": "Fold the top and the four side corners backward as shown in the picture.",
        "image": "images/step4.png"
    },
    {
        "step": 5,
        "instruction": "fin.",
        "image": "images/step5.png"
    }
]


# ---------------------------------------------------------
# ② 手順管理クラス
# ---------------------------------------------------------

class OrigamiTutor:

    def __init__(self, steps):
        self.steps = steps
        self.current_step = 0
        self.finished = False

    # ---------------------------------------------
    # 現在の手順を取得
    # ---------------------------------------------
    def get_current_step(self):

        if self.finished:
            return None

        return self.steps[self.current_step]

    # --------------------------------------------- 
    # CVに渡す現在のステップ番号を取得 
    # --------------------------------------------- 
    def get_current_step_number(self):

        if self.finished: 
            return None 

        return self.steps[self.current_step]["step"]

    # ---------------------------------------------
    # 現在の指示を表示
    # ---------------------------------------------
    def show_instruction(self):

        step = self.get_current_step()

        if step is None:
            print("\n===============================")
            print("Your origami heart is complete!")
            print("===============================")
            return

        print("\n----------------------")
        print(f"Step {step['step']}")
        print(step["instruction"])
        print("----------------------")

    # ---------------------------------------------
    # CV判定を受け取る
    # ---------------------------------------------
    def receive_cv_result(self, result):

        # True = 正しく折れた
        if result:

            print("Correct!")
            self.next_step()

        # False = まだ正しくない
        else:

            print("Not correct yet.")
            print("Please continue the same step.")

    # ---------------------------------------------
    # 次の手順へ
    # ---------------------------------------------
    def next_step(self):

        self.current_step += 1

        if self.current_step >= len(self.steps):

            self.finished = True

            print("\n======================================")
            print("     Your origami heart is complete!")
            print("======================================")

        else:

            print("\nMoving to the next step.")

    # ---------------------------------------------
    # 完成したか
    # ---------------------------------------------
    def is_finished(self):

        return self.finished



def wait_for_cv_result(step):

    # 現在のステップ番号を私、True / Falseの判定結果を受け取る
    
    return check_origami(step)

# ---------------------------------------------------------
# ③ メイン処理
# ---------------------------------------------------------

def main():

    tutor = OrigamiTutor(STEPS)

    print("==============================")
    print("  How to Fold an Origami")
    print("          Heart")
    print("==============================")

    while not tutor.is_finished():

        # 現在の指示を表示
        tutor.show_instruction()

        # 現在のステップ番号を取得
        current_step = tutor.get_current_step_number()

        # -------------------------------------------------
        # ここでCV担当から判定を受け取る
        # -------------------------------------------------

        cv_result = wait_for_cv_result(current_step)

        # CV結果を手順管理に渡す
        tutor.receive_cv_result(cv_result)

        time.sleep(1)


# ---------------------------------------------------------
# ⑤ 実行
# ---------------------------------------------------------

if __name__ == "__main__":
    main()

