# 상위 폴더 경로 찾지 못할 때 사용
import sys
sys.path.append("C:/Users/Lee/Desktop/project/")

import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from Models import *
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split
from actions_recognition_model.Models import TwoStreamSpatialTemporalGraph
from Visualizer import plot_graphs


save_folder = 'saved/TSSTG(pts+mot)-01(cf+hm-hm)'

device = 'cuda'
epochs = 20
batch_size = 32
patience = 8
no_improvement = 0
best_val_loss = 100

data_files = ['C:/Users/Lee/Desktop/project/data_preprocssing/result/pkl/train.pkl',
              'C:/Users/Lee/Desktop/project/data_preprocssing/result/pkl/valid.pkl',
              'C:/Users/Lee/Desktop/project/data_preprocssing/result/pkl/test.pkl',]

class_names = ['Standing', 'Sitting', 'Lying Down','Stand up', 'Sit down', 'Fall Down']
num_class = len(class_names)


def load_dataset(data_files, batch_size, split_size=0):
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return train_loader, valid_loader


def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model


if __name__ == '__main__':
    save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # DATA
    train_loader, _ = load_dataset(data_files[0:1], batch_size)
    valid_loader, train_loader_ = load_dataset(data_files[1:2], batch_size, 0.5)

    train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset, train_loader_.dataset]),
                                   batch_size, shuffle=True)
    dataloader = {'train': train_loader, 'valid': valid_loader}
    del train_loader_

    # MODEL
    graph_args = {'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

    # 최적화
    optimizer = Adadelta(model.parameters())

    losser = torch.nn.BCELoss()

    # 학습(Train + Valid)
    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    for e in range(epochs):
        print('Epoch {}/{}'.format(e, epochs - 1))
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
            else:
                model = set_training(model, False)

            run_loss = 0.0
            run_accu = 0.0
            with tqdm(dataloader[phase], desc=phase) as iterator:
                for pts, lbs in iterator:

                    # 동일한 Node의 좌표의 거리를 입력으로 행동 인지
                    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]

                    mot = mot.to(device)
                    pts = pts.to(device)
                    lbs = lbs.to(device)

                    # Forward
                    out = model((pts, mot))
                    loss = losser(out, lbs)

                    if phase == 'train':
                        # Backward
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()

                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),
                                          lbs.detach().cpu().numpy())
                    run_accu += accu

                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                        loss.item(), accu))
                    

                    iterator.update()
                    # break

            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / len(iterator))
            # break

        

        print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
              ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                             loss_list['valid'][-1], accu_list['valid'][-1]))

        

        # CheckPoint (최적화 모델 저장)
        if loss_list['valid'][-1] < best_val_loss:
            best_val_loss = loss_list['valid'][-1]
            torch.save(model.state_dict(), os.path.join(save_folder, 'CheckPoint.pth'))

        else:
            no_improvement += 1

        # Early Stopping
        if no_improvement >= patience:
            print('Early stopping 발생 (Patience = 8)')
            break


        plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                    '<Loss Graph>', 'Loss', xlim=[0, epochs], ylim=[0.15, 0.3],
                    save=os.path.join(save_folder, 'loss_graph.png'))
        
        plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                    '<Accuracy Graph>', 'Accu', xlim=[0, epochs], ylim=[0.65, 1],
                    save=os.path.join(save_folder, 'accu_graph.png'))

        #break

    del train_loader, valid_loader

    # 최적화 CheckPoint 모델 불러오기
    model.load_state_dict(torch.load(os.path.join(save_folder, 'CheckPoint.pth')))

    # 평가 (Test)
    model = set_training(model, False)
    data_file = data_files[2]
    eval_loader, _ = load_dataset([data_file], 32)

    print('Evaluation.')
    run_loss = 0.0
    run_accu = 0.0
    y_preds = []
    y_trues = []
    with tqdm(eval_loader, desc='eval') as iterator:
        for pts, lbs in iterator:
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(device)
            pts = pts.to(device)
            lbs = lbs.to(device)

            out = model((pts, mot))
            loss = losser(out, lbs)

            run_loss += loss.item()
            accu = accuracy_batch(out.detach().cpu().numpy(),
                                  lbs.detach().cpu().numpy())
            run_accu += accu

            y_preds.extend(out.argmax(1).detach().cpu().numpy())
            y_trues.extend(lbs.argmax(1).cpu().numpy())

            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                loss.item(), accu))
            iterator.update()

    run_loss = run_loss / len(iterator)
    run_accu = run_accu / len(iterator)


    print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))
    
    '''
    plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu{:.4f}'.format(
        os.path.basename(data_file), run_loss, run_accu
    ), 'true', save=os.path.join(save_folder, '{}-confusion_matrix.png'.format(
        os.path.basename(data_file).split('.')[0])))
    '''

   
