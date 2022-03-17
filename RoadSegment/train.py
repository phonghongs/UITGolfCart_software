
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm 
from ex import BiSeNet
from dataset import UTEDataset
from helpers import (
    reverse_one_hot,
    compute_accuracy,
    fast_hist,
    per_class_iu,
    save_checkpoint,
    load_checkpoint
)

EPOCHS = 150
LEARNING_RATE = 0.00001
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 15
CHECKPOINT_STEP = 2
VALIDATE_STEP = 1

NUM_CLASSES = 4
LOAD_MODEL = True

model = BiSeNet(num_classes=NUM_CLASSES, training=True)
model = model.to(device=DEVICE)

#dataloader
dataset_train = UTEDataset(mode='train')
dataset_val = UTEDataset(mode='val')

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

dataloader_val = DataLoader(
    dataset_train,
    batch_size=1,
    shuffle=True
)
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
loss_func = torch.nn.CrossEntropyLoss()

#Validate
def val(model, dataloader):
    accuracy_arr = []
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    with torch.no_grad():
        model.eval()
        print('validating...')

        for i, (val_data, val_label) in enumerate(dataloader):
            val_data = val_data.to(device=DEVICE)
            #output of the model is (1,num_classes,W,H) => (num_classes,W,H)
            val_output = model(val_data).squeeze()
            #convert (nc,W,H) => (W,H) with one hot encoder
            val_output = reverse_one_hot(val_output)
            val_output = np.array(val_output.cpu())
            #Process label and convert to (W,H) image
            val_label = val_label.squeeze()
            val_label = np.array(val_label.cpu())
            #Compute acc and iou
            accuracy = compute_accuracy(val_output, val_label)
            # print('acc?',accuracy)
            hist += fast_hist(val_label.flatten(),val_output.flatten(), NUM_CLASSES)
            #Append to calculate
            accuracy_arr.append(accuracy)

        miou_list = per_class_iu(hist)[:-1]
        mean_accuracy, mean_iou = np.mean(accuracy_arr), np.mean(miou_list)
        print('Mean_accuracy:{} mIOU:{}'.format(mean_accuracy, mean_iou))
        return mean_accuracy, mean_iou

#Training
torch.cuda.empty_cache()
if LOAD_MODEL:
    load_checkpoint(torch.load('checkpoints/best_model.pth'), model)

for epoch in range(EPOCHS):
    model.train()
    tq = tqdm(total=len(dataloader_train) * BATCH_SIZE)
    tq.set_description('Eppch {}/{}'.format(epoch, EPOCHS))

    loss_record = []
    max_iou = 0

    for i, (data, label) in enumerate(dataloader_train):
        data = data.to(device=DEVICE)
        label = label.to(device=DEVICE)
        output, output_sup1, output_sup2 = model(data)
        loss1 = loss_func(output, label)
        loss2 = loss_func(output_sup1, label)
        loss3 = loss_func(output_sup2, label)

        #Combine 3 losses
        loss = loss1 + loss2 + loss3
        tq.update(BATCH_SIZE)
        tq.set_postfix(loss='%6f'%loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    tq.close()
    loss_train_mean = np.mean(loss_record)
    print('loss for train: %f'%(loss_train_mean))

    #save checkpoint
    if epoch % CHECKPOINT_STEP == 0:
        torch.save(model.state_dict(), 'checkpoints/latest_model.pth')

    #validate and save ckp
    if epoch % VALIDATE_STEP == 0:
        _, mean_iou = val(model, dataloader_val)
        if mean_iou > max_iou:
            max_iou = mean_iou
            print('save best model with mIOU = {}'.format(mean_iou))
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, 'checkpoints/best_model.pth')
