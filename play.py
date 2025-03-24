import cv2
import numpy as np
import statistics as st
from collections import deque
from random import choice
from tensorflow.keras.models import Model, load_model


model = load_model("rps1.h5")

# Hàm xác định người thắng cuộc
def findout_winner(user_move, computer_move):
    if user_move == computer_move:
        return "Tie"
    if (user_move == "rock" and computer_move == "scissor") or \
       (user_move == "paper" and computer_move == "rock") or \
       (user_move == "scissor" and computer_move == "paper"):
        return "User"
    return "Computer"

# Hàm hiển thị nước đi của máy tính
def display_computer_move(move, frame):
    text = f"Computer chose: {move}"
    cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Hàm hiển thị kết quả và hỏi người chơi có muốn chơi lại không

def show_winner(user_score, computer_score):
    print(f"User Score: {user_score}, Computer Score: {computer_score}")
    if user_score > computer_score:
        print("🎉 Bạn thắng!")
    elif user_score < computer_score:
        print("😞 Máy thắng!")
    else:
        print("🤝 Hòa!")
    
    play_again = input("Chơi lại không? (y/n): ").strip().lower()
    return play_again == 'y'

# Mở webcam
cap = cv2.VideoCapture(0)
box_size = 234
width = int(cap.get(3))

# Thiết lập số lần chơi
attempts = 5
total_attempts = attempts

# Biến lưu kết quả
computer_move_name = "nothing"
final_user_move = "nothing"
label_names = ['nothing', 'paper', 'rock', 'scissor']
computer_score, user_score = 0, 0
rect_color = (255, 0, 0)
hand_inside = False
confidence_threshold = 0.70
smooth_factor = 5
de = deque(['nothing'] * 5, maxlen=smooth_factor)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

    # Xác định vùng tay
    roi = frame[5: box_size-5, width-box_size + 5: width - 5]
    roi = np.array([roi]).astype('float64') / 255.0

    # Mô hình dự đoán
    pred = model.predict(roi)  # <--- Đảm bảo đã load mô hình trước đó
    move_code = np.argmax(pred[0])
    user_move = label_names[move_code]
    prob = np.max(pred[0])

    # Nếu xác suất đủ cao, cập nhật kết quả
    if prob >= confidence_threshold:
        de.appendleft(user_move)
        final_user_move = max(set(de), key=de.count)  # Sử dụng mode() thủ công

        if final_user_move != "nothing" and not hand_inside:
            hand_inside = True
            computer_move_name = choice(['rock', 'paper', 'scissor'])
            winner = findout_winner(final_user_move, computer_move_name)
            display_computer_move(computer_move_name, frame)
            total_attempts -= 1

            if winner == "Computer":
                computer_score += 1
                rect_color = (0, 0, 255)
            elif winner == "User":
                user_score += 1
                rect_color = (0, 255, 0)
            else:
                rect_color = (255, 255, 255)

            # Nếu hết lượt, hỏi người chơi có muốn tiếp tục không
            if total_attempts == 0:
                play_again = show_winner(user_score, computer_score)
                if play_again:
                    user_score, computer_score, total_attempts = 0, 0, attempts
                else:
                    break

        elif final_user_move != "nothing":
            display_computer_move(computer_move_name, frame)
        else:
            hand_inside = False
            rect_color = (255, 0, 0)

    # Hiển thị điểm số
    cv2.putText(frame, f"Your Move: {final_user_move}", (420, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Computer's Move: {computer_move_name}", (2, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Your Score: {user_score}", (420, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Computer Score: {computer_score}", (2, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Attempts left: {total_attempts}", (190, 400), 
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 2, 255), 1)
    
    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), rect_color, 2)
    cv2.imshow("Rock Paper Scissors", frame)

    # Thoát nếu nhấn 'q'
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
