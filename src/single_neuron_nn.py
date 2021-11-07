import torch
import MinkowskiEngine as ME

from utils import to_minkowski_tensor
from simple_reader import LigandDataset, DataLoader

class MinkowskiNN(ME.MinkowskiNetwork):
    def __init__(
        self
    ):
        ME.MinkowskiNetwork.__init__(self, 3)

        self.conv = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            dimension=self.D
        )
        self.pool = ME.MinkowskiGlobalSumPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.conv(x)
        x = self.pool(x)
        return x.F.squeeze(0).squeeze(0)

model = MinkowskiNN()
dataset = LigandDataset('data')
dataloader = DataLoader(dataset)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for idx, (blob, y) in enumerate(dataloader):
    if idx >= 10000:
        break
        
    optimizer.zero_grad()
    y = blob.sum()
    blob = to_minkowski_tensor(blob)
    y_hat = model(blob)
    loss = criterion(y, y_hat)
    loss.backward()
    optimizer.step()

    if not idx % 100:
        print(f'iteration:{idx:>8}', f'loss: {loss.item():.4f}', f'groundtruth: {y.item():6.4f}', f'prediction: {y_hat.item():.4f}')
    
    if not idx % 500:
        print([param for param in model.parameters()]) 

print([param for param in model.parameters()])
