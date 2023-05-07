import gensim.downloader as api
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import *

with open('spo_train_cleaned_1.txt', 'r') as f:
    first_list_train = []
    fourth_list_train = []
    fifth_list_train = []
    for line in f:
        row = line.strip().split('\t')
        if (row[3] is not None) and (row[4] is not None):
            row_3 = eval(row[3])
            row_4 = eval(row[4])
            if (row_3 is not None) and (row_4 is not None):
                if (len(row_3) == 3) and (len(row_4) == 3) and (None not in (row_3)) and (None not in (row_4)):
                    lower_row_3 = []
                    for i in row_3:
                        lower_row_3.append(i.lower())
                    lower_row_4 = []
                    for i in row_4:
                        lower_row_4.append(i.lower())
                    first_list_train.append(int(row[0]))
                    fourth_list_train.append(lower_row_3)
                    fifth_list_train.append(lower_row_4) 

with open('spo_test_cleaned_1.txt', 'r') as f:
    first_list_test = []
    fourth_list_test = []
    fifth_list_test = []
    for line in f:
        row = line.strip().split('\t')
        if (row[3] is not None) and (row[4] is not None):
            row_3 = eval(row[3])
            row_4 = eval(row[4])
            if (row_3 is not None) and (row_4 is not None):
                if (len(row_3) == 3) and (len(row_4) == 3) and (None not in (row_3)) and (None not in (row_4)):
                    lower_row_3 = []
                    for i in row_3:
                        lower_row_3.append(i.lower())
                    lower_row_4 = []
                    for i in row_4:
                        lower_row_4.append(i.lower())
                    first_list_test.append(int(row[0]))
                    fourth_list_test.append(lower_row_3)
                    fifth_list_test.append(lower_row_4)


model = api.load("glove-twitter-50")


# List of words

fourth_list_tensor_train = []
fifth_list_tensor_train = []
new_label_train = []

for i in range(0, len(fourth_list_train)):
    try:
        vectors1 = [model[word1] for word1 in fourth_list_train[i]]
        numpy_array1 = np.array(vectors1)
        tensor1 = torch.tensor(numpy_array1)
        

        vectors2 = [model[word2] for word2 in fifth_list_train[i]]
        numpy_array2 = np.array(vectors2)
        tensor2 = torch.tensor(numpy_array2)
        
    except:
        continue

    fourth_list_tensor_train.append(tensor1)
    fifth_list_tensor_train.append(tensor2)
    new_label_train.append(float(first_list_train[i]))




print(len(fourth_list_tensor_train))
print(len(fifth_list_tensor_train))

fourth_list_tensor_test = []
fifth_list_tensor_test = []
new_label_test = []

for i in range(0, len(fourth_list_test)):
    try:
        vectors1 = [model[word1] for word1 in fourth_list_test[i]]
        numpy_array1 = np.array(vectors1)
        tensor1 = torch.tensor(numpy_array1)
        

        vectors2 = [model[word2] for word2 in fifth_list_test[i]]
        numpy_array2 = np.array(vectors2)
        tensor2 = torch.tensor(numpy_array2)
        
    except:
        continue

    fourth_list_tensor_test.append(tensor1)
    fifth_list_tensor_test.append(tensor2)
    new_label_test.append(float(first_list_test[i]))


print(len(fourth_list_tensor_test))
print(len(fifth_list_tensor_test))

# Convert lists to tensors
x1_train = torch.stack(fourth_list_tensor_train)
x2_train = torch.stack(fifth_list_tensor_train)
y_train = torch.tensor(new_label_train)

# Create a TensorDataset from x1, x2, and y
train_dataset = TensorDataset(x1_train, x2_train, y_train)

# Convert lists to tensors
x1_test = torch.stack(fourth_list_tensor_test)
x2_test = torch.stack(fifth_list_tensor_test)
y_test = torch.tensor(new_label_test)

# Create a TensorDataset from x1, x2, and y
test_dataset = TensorDataset(x1_test, x2_test, y_test)

# Define the batch size
batch_size = 64

# Create a DataLoader from the TensorDataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
for i, (x1, x2, y) in enumerate(train_loader):
    print(x1.unsqueeze(1).shape)
    print(x2.unsqueeze(1).shape)

#DEFINE CNN MODEL

# Define the input shape
input_shape = (3, 50)

# Define the number of filters
filters = 1

# Define the kernel size
kernel_size = 3

# Define the number of units in the dense layer
units = 10

# Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.fc1 = nn.Linear(96, 50)
        #self.fc2 = nn.Linear(16*24*14, 128)

    def forward(self, x1, x2):
        # Pass sentence 1 through the network
        x1 = x1.unsqueeze(1)
        print(x1.shape)
        x1 = self.conv1(x1)
        print(x1.shape)
        x1 = F.relu(x1)
        print(x1.shape)
        x1 = self.pool1(x1)
        print(x1.shape)
        x1 = self.conv2(x1)
        print(x1.shape)
        x1 = F.relu(x1)
        print(x1.shape)
        x1 = self.pool2(x1)
        print(x1.shape)
        x1 = torch.flatten(x1, 1) # Flatten the output
        x1 = self.fc1(x1)

        # Pass sentence 2 through the network
        x2 = x2.unsqueeze(1)
        print(x2.shape)
        x2 = self.conv1(x2)
        print(x2.shape)
        x2 = F.relu(x2)
        print(x2.shape)
        x2 = self.pool1(x2)
        print(x2.shape)
        x2 = self.conv2(x2)
        print(x2.shape)
        x2 = F.relu(x2)
        print(x2.shape)
        x2 = self.pool2(x2)
        print(x2.shape)
        #x2 = x2.view(-1, 6144) # Flatten the output
        x2 = torch.flatten(x2, 1)
        x2 = self.fc1(x2)

        # Calculate Manhattan distance
        d = torch.abs(x1 - x2).sum(dim=1)

        # Calculate similarity score using e^-d
        score = torch.exp(-d)

        return score

# Create an instance of the model
model = CNN()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


n_epochs = 1

def train(model, train_loader, criterion=criterion, optimizer=optimizer, n_epoch=n_epochs):

    model.train()

    for epoch in range(n_epoch):
        running_loss = 0
        for i, (x1, x2, y) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x1, x2)
            print(outputs.shape)
            print(y.shape)
            print(outputs.dtype)
            print(y.dtype)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

    return model

model = train(model, train_loader)

def eval_model(model, dataloader):
    
    model.eval()
    Y_pred = []
    Y_true = []
    for i, (x1, x2, y) in enumerate(val_loader):
        # your code here
        predicted = model(x1, x2)
        for element in predicted:
            Y_pred.append(element.item())
        for element in y:
            Y_true.append(element.item())
    #Y_pred = np.concatenate(Y_pred, axis=0)
    #Y_true = np.concatenate(Y_true, axis=0)

    return Y_pred, Y_true

y_pred, y_true = eval_model(model, val_loader)

for i in range(len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1.0
    else:
        y_pred[i] = 0.0

acc = accuracy_score(y_true, y_pred)
f1score = f1_score(y_true, y_pred)
print(acc)
print(f1score)
    
