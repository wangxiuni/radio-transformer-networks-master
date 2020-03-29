import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
NUM_EPOCHS = 17
BATCH_SIZE = 256
CHANNEL_SIZE = 4
USE_CUDA = True
DOUBLE_N = 7
M = 2**CHANNEL_SIZE
# bit / channel_use
communication_rate = 4/7

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RadioTransformerNetwork(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(RadioTransformerNetwork, self).__init__()

        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
            nn.BatchNorm1d(compressed_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_dim, in_channels)
        )

    def forward(self, x):
        x = self.encoder(x)

        # Normalization.
        #x = (self.in_channels ** 2) * (x / x.norm(dim=-1)[:, None])

        # 7dBW to SNR.
        training_signal_noise_ratio = 5.01187

        # Simulated Gaussian noise.
        noise = torch.autograd.Variable(torch.randn(*x.size()) / math.sqrt(2 * communication_rate * training_signal_noise_ratio))
        if USE_CUDA: noise = noise.cuda()
        x += noise

        x = self.decoder(x)

        return x


class Encoder(nn.Module):
    def __init__(self,in_channels, compressed_dim):
        super(Encoder, self).__init__()

        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
            nn.BatchNorm1d(compressed_dim)
        )


    def forward(self, x):
        x = self.encoder(x)

        # Normalization.
        #x = (self.in_channels ** 2) * (x / x.norm(dim=-1)[:, None])

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_dim, in_channels)
        )

    def forward(self, x):
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
    import numpy as np
    import torch.optim as optim
    model = RadioTransformerNetwork(M, compressed_dim=DOUBLE_N)
    if USE_CUDA: model = model.cuda()

    train_labels = (torch.rand(10000) * M).long()
    train_data = torch.sparse.torch.eye(M).index_select(dim=0, index=train_labels)

    test_labels = (torch.rand(45000) * M).long()
    test_data = torch.sparse.torch.eye(M).index_select(dim=0, index=test_labels)

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

    class_correct = list(0. for i in range(M))
    class_total = list(0. for i in range(M))
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

    for i in range(M):
        print('Accuracy of %5s : %2d %2d ' % (
            i, class_correct[i], class_total[i]))

    PATH = './E2E.pth'
    torch.save(model.state_dict(), PATH)

    encoder = Encoder(M, DOUBLE_N)
    encoder.load_state_dict(torch.load(PATH), strict=False)
    #encoder.cuda()

    decoder = Decoder(M, DOUBLE_N)
    decoder.load_state_dict(torch.load(PATH), strict=False)
    #decoder.cuda()

    EbNodB_range = list(i for i in np.arange(-4.0, 8.5, 0.5))
    #print(EbNodB_range)
    ber = [None] * len(EbNodB_range)

    for n in range(len(EbNodB_range)):
        EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
        noise_std = np.sqrt(1 / (2 * communication_rate * EbNo))
        all_errors = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data['signal'], data['label']

                encoded_signal = encoder(inputs)
                noise = torch.autograd.Variable(torch.randn(*encoded_signal.size()) / ((2 * communication_rate *EbNo) ** 0.5))
                #print(encoded_signal.shape, len(noise))
                final_signal = encoded_signal + noise
                final_signal = final_signal.float()
                #print(encoded_signal, final_signal)
                #print(final_signal.dtype)
                _, outputs = torch.max(decoder(final_signal), 1)
                errors = (outputs != labels)
                errors = errors.numpy().sum()
                all_errors += errors

        ber[n] = all_errors/45000
        print("SNR:", EbNodB_range[n], "BER:", ber[n])

    import matplotlib.pyplot as plt

    plt.plot(EbNodB_range, ber, 'bo', label='Autoencoder(7,4)')
    # plt.plot(list(EbNodB_range), ber_theory, 'ro-',label='BPSK BER')
    plt.yscale('log')
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.legend(loc='upper right', ncol=1)

    plt.savefig('AutoEncoder_7_4_BER_matplotlib')
    plt.show()