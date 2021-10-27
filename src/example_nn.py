import torch
import MinkowskiEngine as ME

from utils import to_me_tensor
from simple_reader import LigandDataset, DataLoader

class MinkowskiCNN(ME.MinkowskiNetwork):
    def __init__(
        self
    ):
        ME.MinkowskiNetwork.__init__(self, 3)

        self.conv = ME.MinkowskiConvolution(
            in_channels=1, 
            out_channels=1,
            kernel_size=3,
            dimension=self.D
        )
        self.pool = ME.MinkowskiGlobalSumPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.conv(x)
        x = self.pool(x)
        return x

model = MinkowskiCNN()
dataset = LigandDataset('data')
dataloader = DataLoader(dataset)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for idx, (blob, label) in enumerate(dataloader):
    if idx >= 10000:
        break

    optimizer.zero_grad()

    y = blob.sum() # neural networks objective is to compute sum of all densities stored in the blob
    x = to_me_tensor(blob.unsqueeze(0)) # (for now) dataloader yield torch tensor, here we convert single blob to minibatch containing single Minkowski sparse tensor
    y_hat = model(x).F # forward pass

    loss = criterion(y, y_hat)
    loss.backward()
    optimizer.step()

    if not idx % 100:
        #print(loss.grad)
        #print([param.grad for param in model.parameters()]) 
        print(f'iteration:{idx:>8}', f'loss: {loss.item():.4f}', f'groundtruth: {y.item():6.4f}', f'prediction: {y_hat.item():.4f}')