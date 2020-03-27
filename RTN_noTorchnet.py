import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 100
BATCH_SIZE = 256
CHANNEL_SIZE = 4
USE_CUDA = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RadioTransformerNetwork(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(RadioTransformerNetwork, self).__init__()

        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_dim, in_channels)
        )

    def forward(self, x):
        x = self.encoder(x)

        # Normalization.
        x = (self.in_channels ** 2) * (x / x.norm(dim=-1)[:, None])

        # 7dBW to SNR.
        training_signal_noise_ratio = 5.01187

        # bit / channel_use
        communication_rate = 1

        # Simulated Gaussian noise.
        noise = torch.autograd.Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if USE_CUDA: noise = noise.cuda()
        x += noise

        x = self.decoder(x)

        return x


class TensorDataset(Dataset):

    def __init__(self, data_name, label_name,transform = None):
        self.data_all = data_name
        print(self.data_all.shape)

        self.label_all = label_name
        print(self.label_all.shape)

        self.transform = transform

    def __len__(self):
        return len(self.label_all)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data_all[idx,:]
        label = self.label_all[idx]

        sample = {'signal':data, 'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    import math
    import torch.optim as optim
    model = RadioTransformerNetwork(CHANNEL_SIZE, compressed_dim=int(math.log2(CHANNEL_SIZE)))
    if USE_CUDA: model = model.cuda()

    train_labels = (torch.rand(10000) * CHANNEL_SIZE).long()
    train_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels)

    test_labels = (torch.rand(1500) * CHANNEL_SIZE).long()
    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    trainset = TensorDataset(train_data, train_labels)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)

    testset = TensorDataset(test_data, test_labels)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)

    Loss_list = []
    Accuracy_list = []

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['signal'], data['label']

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)#.squeeze()
            c = c.cpu()
            c = c.sum()
            c = c.numpy()

            Accuracy_list.append(c / BATCH_SIZE) if i % 40 != 39 else Accuracy_list.append(c / 16)

            Loss_list.append(loss.item() / BATCH_SIZE)

            if i % 40 == 39:
                print('[%d, %5d] loss: %.3f acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 39, Accuracy_list[-2]))
                running_loss = 0.0

    print('Finished Training')

    class_correct = list(0. for i in range(CHANNEL_SIZE))
    class_total = list(0. for i in range(CHANNEL_SIZE))
    with torch.no_grad():
        for data in testloader:
            # print(data.shape)
            inputs, labels = data['signal'], data['label']

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            c = (predicted == labels)
            c = c.cpu()
            #c = c.numpy()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                if c[i].item() != True:
                    print(label, predicted[i])

    for i in range(CHANNEL_SIZE):
        print('Accuracy of %5s : %2d %2d ' % (
            i, class_correct[i], class_total[i]))
