from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import PIL

# hyperparameters of model
# as global variables

epochs = 1
batch_size = 30
predict_batch_size = 1
learning_rate = 0.03
l2_decay = 0
momentum = 0.8


class LandmarkDataset(Dataset):
  # our landmark dataset that we define
  # to process, transform and clean the data
    def __init__(self, dataset, labels, transform=None, isTestModel=False):
        self.labels = labels
        self.dataset = dataset
        #print ("type(self.labels) :", type(self.labels))
        #print ("type(self.dataset) :", type(self.dataset))
        # initialize transformation of data (default is None)
        self.transform = transform
        self.isTestModel = isTestModel

    def __len__(self):
        # 'Denotes the total number of samples'

        return len(self.dataset)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        #print ("type(self.dataset) :", type(self.dataset))
        ID = self.dataset.iloc[index]
        #print ("ID :", ID)
        #print ("ID type :", type(ID))
        # Load data and get label
        x = PIL.Image.open('hw7data/images/' + ID + '.jpg')
        if self.transform != None: x = self.transform(x)
        #print ("type(self.labels) :", type(self.labels))
        if self.isTestModel: y = 0
        elif not self.isTestModel: y = self.labels.loc[ID][0]
        #print ("x", x)
        #print ("y", y)
        return x, y

# data augmentation and normalization for training

transform = {
    'train' : transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    'predict' : transforms.Compose(
        [transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

}

"""
loading phase of neural net left omitted from main()
    # load trained model
    model = models.resnet50(pretrained=False)
    model._modules['fc'] = nn.Linear(2048, 10)
    model.load_state_dict(torch.load('trained_resnet50.pt'))
    model.eval()
"""

 ## training phase left out of main for syntax reasons
"""
    for epoch in range(epochs):
        print("Epoch {} / {}".format(epoch+1,epochs))
        t_set_len = len(train_set)
        train_count = 0
        for (x,y) in train_dataLoad:
            train_count += batch_size
            print ("train_count / t_set_len : {} / {}".format(train_count,t_set_len))
            predicted_y = pred_model(x)
            loss_val = loss_function(predicted_y,y)
            opt.zero_grad()
            loss_val.backward()
            print ("loss_val :", loss_val)
            opt.step()
        print ("final loss_val after epoch: ", loss_val)
        if (epoch == epochs - 1): #final epoch
            # save model in a file in the same folder path
            torch.save(pred_model.state_dict(), 'resnet34.pt')
"""

def main():
    train_file = pd.read_csv("hw7data/train.csv")
    train_set = train_file.iloc[:,1]
    t_label_temp = train_file.iloc[:,[1,-1]]
    train_labels = t_label_temp.set_index('id')
    test_set = pd.read_csv("hw7data/test.csv").iloc[:,1]
    #print ("train_set :", train_set)
    #print ("train_labels", train_labels)
    print ("test_set :", test_set)

    train_dataset = LandmarkDataset(train_set,
                                    train_labels,transform=transform['train'])
    train_dataLoad = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    loss_function = nn.NLLLoss()
    pred_model = models.resnet50(pretrained=True)
    pred_model._modules['fc'] = nn.Linear(2048, 10)

    opt = torch.optim.SGD(pred_model.parameters(), lr=learning_rate,
                         momentum=momentum, weight_decay=l2_decay)

    # now train our resnet model with negative log likelihood loss function
    # on 2 epochs with our hyperparameters specified above
    # and run backpropogation with stochastic gradient descent

    model = models.resnet50(pretrained=False)
    model._modules['fc'] = nn.Linear(2048, 10)
    model.load_state_dict(torch.load('trained_resnet50.pt'))
    model.eval()


    empty_labels = pd.DataFrame([])
    # now generalize trained model to test set
    # for instance in test_dataset
    test_dataset = LandmarkDataset(test_set,empty_labels,
                                   transform=transform['predict'],
                                   isTestModel=True)
    test_dataLoad = DataLoader(test_dataset,batch_size=predict_batch_size,
                               shuffle=False)


    # run on test set
    predict_submission = open("submission.txt", "w")
    predict_submission.write("landmark_id\n")
    test_set_count = 0
    len_testdata = len(test_set)
    for (x,y) in test_dataLoad:
        test_set_count += 1
        print ("test_set_count :", test_set_count)
        test_y_val = str(int(torch.argmax(model(x))))
        predict_submission.write(test_y_val + "\n")
    return None

if __name__ == "__main__":
    main()
