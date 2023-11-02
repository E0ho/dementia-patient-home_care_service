# 각 Video Frame별 Action Label 부여
import os
import cv2
import numpy as np
import pandas as pd

# Action 라벨
class_names = ['Standing', 'Sitting','Lying Down', 'Stand up', 'Sit down', 'Fall Down']


# Video 경로
video_folder = 'C:/Users/Lee/Desktop/project/Data/Videos'
# 저장 파일 경로
annot_file_2 = 'C:/Users/Lee/Desktop/project/Data/Action_Label.csv'



video_list = sorted(os.listdir(video_folder))
cols = ['video', 'frame', 'label']
df = pd.DataFrame(columns=cols)


for index_video_to_play in range(len(video_list)):

    video_file = os.path.join(video_folder, video_list[index_video_to_play])
    print(os.path.basename(video_file))

    cap = cv2.VideoCapture(video_file)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video = np.array([video_list[index_video_to_play]] * frames_count)
    frame_1 = np.arange(1, frames_count + 1)
    label = np.array([0] * frames_count)

    k = 0
    i = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        label[i-1] =k
        if ret:

            cls_name = class_names[k]

            # Lable 보조 텍스트 활용
            frame = cv2.resize(frame, (0, 0), fx=3, fy=3)
            frame = cv2.putText(frame, 'Video: {}     Total_frames: {}        Frame: {}       Pose: {} '.format(video_list[index_video_to_play],frames_count,i+1, cls_name,),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            frame = cv2.putText(frame, 'Back:  a',(10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            frame = cv2.putText(frame,'Standing:   0', (10, 300),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Sitting:    1', (10, 330),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Lying Down: 2', (10, 360),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Stand Up:   3', (10, 390),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Sit Down:   4', (10, 420),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Fall Down:  5', (10, 450),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 화면 출력
            cv2.imshow('Action Label', frame)
            key = cv2.waitKey(0) & 0xFF

            
            # key를 통한 Label 부여
            if key == ord('q'):
                break
            elif key == ord('0'): # Standing
                i += 1
                k=0
            elif key == ord('1'): # Sitting
                i += 1
                k = 1
            elif key == ord('2'): # Lying down
                i += 1
                k = 2
            elif key == ord('3'): # Stand Up
                i += 1
                k = 3
            elif key == ord('4'): # Sit Down
                i += 1
                k = 4
            elif key == ord('5'): # Fall Down
                i += 1
                k = 5
            elif key == ord('a'): # Back
                i -= 1
        else:
            break
    rows = np.stack([video, frame_1, label], axis=1)
    df = df.append(pd.DataFrame(rows, columns=cols),ignore_index=True)
df.to_csv(annot_file_2,index=False)
cap.release()
cv2.destroyAllWindows()