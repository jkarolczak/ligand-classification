import torch
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from utils import *
from simple_reader import LigandDataset

import gc

import neptune.new as neptune

class PoCMinkNet(ME.MinkowskiNetwork):
    """A class to represent MinkNet neural network."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int = 3
    ):
        """
        :param conv_channels: list of integers describing consecutive convolution
        kernels sizes - it is assumed that convolution kernel is a cube, all 
        kernel dimensions are equal
        :param in_channels: number of channels in input 
        :param out_channels: number of channels in output, equal to number of classes
        :param dimensions: number of dimensions of input
        """
        ME.MinkowskiNetwork.__init__(self, dimensions)
        self.sparse_conv_blocks = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=4,
                kernel_size=3,
                dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=4,
                out_channels=4,
                kernel_size=3,
                dimension=self.D
            ),
            ME.MinkowskiTanh(),
            ME.MinkowskiMaxPooling(
                kernel_size=2,
                dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                dimension=self.D
            ),
            ME.MinkowskiTanh(),
            ME.MinkowskiMaxPooling(
                kernel_size=2,
                dimension=self.D
            )
        )

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.linear1 = torch.nn.Linear(
            in_features=2 * 8,
            out_features=8
        )
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(
            in_features=8, 
            out_features=out_channels
        )
        self.softmax = torch.nn.Softmax(-1)
        

    def forward(
        self, 
        x: ME.SparseTensor
    ) -> torch.Tensor:
        
        x = self.sparse_conv_blocks(x)
    
        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)

        x = torch.cat([x_avg.F, x_max.F], -1).squeeze(0)
        
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.softmax(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return x
    

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    run = neptune.init(
        project="LIGANDS/LIGANDS",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMGQ1ZDQwZS05YjhlLTRmMGUtYjZjZC0yYzk0OWE4OWJmYzkifQ==",
    ) 
    
    dataset_path = 'data/labels_three.csv'
    dataset = LigandDataset('data', dataset_path)

    train, test = dataset_split(dataset=dataset)

    train_dataloader = DataLoader(
        dataset=train, 
        batch_size=1, 
        collate_fn=collation_fn,
        num_workers=4,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test, 
        batch_size=1, 
        collate_fn=collation_fn,
        num_workers=4,
        shuffle=True
    )

    model = PoCMinkNet(
        in_channels = 1,
        out_channels = dataset.labels[0].shape[0]
    )
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3
    )
    epochs = 10

    log_config(
        run=run,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataset=dataset
    )

    for e in range(epochs):
        model.train()
        for idx, (coords, feats, labels) in enumerate(train_dataloader):
            
            labels = labels.to(device=device)
            batch = ME.SparseTensor(feats, coords, device=device)

            optimizer.zero_grad()
            labels_hat = model(batch)
            loss = criterion(labels_hat, labels)
            
            loss.backward()

            optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del batch, labels, labels_hat
            gc.collect()
            
        model.eval()
        with torch.no_grad():
            groundtruth, predictions = None, None
            for idx, (coords, feats, labels) in enumerate(test_dataloader):
                torch.cuda.empty_cache()
                batch = ME.SparseTensor(feats, coords, device=device)
                preds = model(batch)
                if groundtruth is None:
                    groundtruth = labels
                    predictions = preds
                else:
                    try:
                        groundtruth = torch.cat([groundtruth, labels], 0)
                        predictions = torch.cat([predictions, preds], 0)
                    except:
                        pass 

        log_state_dict(
            run=run,
            model=model
        )
        log_epoch(
            run=run,
            preds=predictions, 
            target=groundtruth
        )
        
    run.stop()
    model.eval()
    with torch.no_grad():
        groundtruth, predictions = None, None
        for idx, (coords, feats, labels) in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            batch = ME.SparseTensor(feats, coords, device=device)
            preds = model(batch)
            if groundtruth is None:
                groundtruth = labels
                predictions = preds
            else:
                try:
                    groundtruth = torch.cat([groundtruth, labels], 0)
                    predictions = torch.cat([predictions, preds], 0)
                except:
                    pass 
    