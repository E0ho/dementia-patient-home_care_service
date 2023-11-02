# Bounding Box 정보를 이용한 Skeleton KeyPoints 추출
import os
import cv2
import time
import torch
import pandas as pd
import numpy as np


# 상위 폴더 경로 찾지 못할 때 사용
import sys
sys.path.append("C:/Users/Lee/Desktop/project/")


from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast


# 저장 경로
save_path = 'C:/Users/Lee/Desktop/project//Data/Skeleton_KeyPoints.csv'

# 활용 파일
video_folder = 'C:/Users/Lee/Desktop/project//Data/Videos'            # 비디오 경로

Action_Label_Annotation_file = 'C:/Users/Lee/Desktop/project//Data/Action_Label.csv'  # Action Label Annotation 파일
Bounding_Box_Annotation_file = 'C:/Users/Lee/Desktop/project//Data/Annotation_Files'  # Bounding Box Annotation 파일



# Object Detection 적용
detector = TinyYOLOv3_onecls()

# Pose Estimation 적용
inp_h = 320
inp_w = 256
pose_estimator = SPPE_FastPose('resnet101', 'resnet101')

class_names = ['Standing', 'Sitting', 'Lying Down','Stand up', 'Sit down', 'Fall Down']

# 추출할 KeyPotint 정보
columns = ['video', 'frame', 
           'Nose_x', 'Nose_y', 'Nose_s',
           'LShoulder_x', 'LShoulder_y', 'LShoulder_s','RShoulder_x', 'RShoulder_y', 'RShoulder_s', 
           'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x','RElbow_y', 'RElbow_s', 
           'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 
           'LKnee_x', 'LKnee_y', 'LKnee_s','RKnee_x', 'RKnee_y', 'RKnee_s', 
           'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y','RAnkle_s', 
           'label']


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy

# Video Frame별 Action Label
annot = pd.read_csv(Action_Label_Annotation_file)
vid_list = annot['video'].unique()

# 각 Frame 가져오기
for vid in vid_list:
    print(f'Process on: {vid}')
    df = pd.DataFrame(columns=columns)
    cur_row = 0

    # Action Labels 정보 읽기
    frames_label = annot[annot['video'] == vid].reset_index(drop=True)

    cap = cv2.VideoCapture(os.path.join(video_folder, vid))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Bounding Boxs Labels 정보 읽기
    annot_file_2 = os.path.join(Bounding_Box_Annotation_file, vid.split('.')[0])
    annot_file_2=annot_file_2+'.txt'
    annot_2 = []

    # Bounding Boxs 정보가 있을 때만 읽기
    if os.path.exists(annot_file_2):
        annot_2 = pd.read_csv(annot_file_2, header=None,
                                  names=['frame_idx', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        annot_2 = annot_2.dropna().reset_index(drop=True)


    fps_time = 0
    i = 1
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cls_idx = int(frames_label[frames_label['frame'] == i]['label'])

            if len(annot_2):
                bb = np.array(annot_2.iloc[i-1, 2:].astype(int))
            else:
                bb = detector.detect(frame)
                if bb == None:
                    bb = torch.tensor([[0, 0, 0, 0, 0.9330, 1.0000, 0.0000]])
                    bb = bb[0, :4].numpy().astype(int)
                else:
                    bb = bb[0, :4].numpy().astype(int)
            bb[:2] = np.maximum(0, bb[:2] - 5)
            bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]

            result = []
            if bb.any() != 0:
                result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                                torch.tensor([[1.0]]))

            if len(result) > 0:
                pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                     frame_size[0], frame_size[1])
                pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)

                #idx = result[0]['kp_score'] <= 0.05
                #pt_norm[idx.squeeze()] = np.nan
                row = [vid, i, *pt_norm.flatten().tolist(), cls_idx]
                scr = result[0]['kp_score'].mean()
            else:
                row = [vid, i, *[np.nan] * (13 * 3), cls_idx]
                scr = 0.0

            df.loc[cur_row] = row
            cur_row += 1

            # 정보 추출 과정 시각화
            frame = vis_frame_fast(frame, result)
            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (125, 255,255 ), 2)
            frame = cv2.putText(frame,vid+'   Frame:' +str(i),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0 ), 2)
            frame = cv2.putText(frame, 'Pose: {}, Score: {:.4f}'.format( class_names[cls_idx], scr),
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 2)
            if not len(annot_2):
                frame = cv2.putText(frame, 'No annotation',
                                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            frame = frame[:, :, ::-1]
            fps_time = time.time()
            i += 1

            frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
            cv2.imshow('frame', frame)
            key =cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('p'):
                key =cv2.waitKey(0)
                if key & 0xFF == ord('q'):
                    break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)