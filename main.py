from dataset import MolecularDataset
from unet import Net
import numpy as np
import torch
import torch.optim as optim
from ground_truth import contstuct # NOT IMPLEMENTED YET!

data_dir = "Data/"
dataset = MolecularDataset(data_dir)

N_before = np.sum( dataset.no_electrons() )

dataset.clean()

N_after = np.sum( dataset.no_electrons() )

print("Number of electrons before cut: %.5f" %N_before)
print("Number of electrons after cut: %.5f" %N_after)
print("Total removed electrons: %.5f" %(N_before - N_after) )


#Using CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Compute device: ")
print(device)

net = Net(8)
net.to(device)

# Preparing data
input = np.stack( dataset.systems['data'].values )
true = construct(dataset)

input = torch.tensor(input[:, np.newaxis, :, :, :]).to(device)


# Training the neural network
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(input):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(data)
        loss = criterion(output, true[i])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')