from sklearn.model_selection import train_test_split
from base_experiment import BaseExperiment
from torch import nn, optim
import torch

class MultiLayerPerceptronModel(nn.Module):
    def __init__(self, params, in_dim, out_dim):
        super().__init__()
        mid_dim = params['mid_dim']
        self.layers = nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, out_dim),
                nn.Sigmoid()
                )
        self.params=params
        self.train_loss = None
        self.train_conf = None
        self.val_loss = None
        self.val_conf = None


    def forward(self, test_emb, np_input=True):
        x = torch.from_numpy(test_emb) if np_input else test_emb
        x = self.layers(x)
        return x if not np_input else (x.detach().numpy(), {'train_loss': self.train_loss, 'train_conf': self.train_conf})


class MultiLayerPerceptronExperiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, params, train_data, early_stopping_patience=10):
        train_acc, train_emb = torch.from_numpy(train_data[0]), torch.from_numpy(train_data[1])

        # split train into train and val
        train_acc, val_acc, train_emb, val_emb = train_test_split(train_acc, train_emb, test_size=0.2, random_state=42)

        model = MultiLayerPerceptronModel(params, train_emb.shape[1], train_acc.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        loss_func = nn.BCELoss()

        train_loss = []
        train_conf = []

        val_loss = []
        val_conf = []

        best_val_loss = float("inf")
        epoch_no_improv = 0

        for epoch in range(params['num_epochs']):

            # training step
            model.train()
            scores = model(train_emb, np_input=False)
            loss = loss_func(scores, train_acc.type(torch.float32))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss.append(loss.detach().item())
            conf = (scores > 0.5) == train_acc
            train_conf.append(conf.double().mean().detach().item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train loss: {train_loss[-1]}, Train confidence: {train_conf[-1]}")

            # validation step
            model.eval()
            with torch.no_grad():
                val_scores = model(val_emb, np_input=False)
                loss_val = loss_func(val_scores, val_acc.type(torch.float32))
                val_loss.append(loss_val.item()) # loss_val.detach().item()

                conf_val = (val_scores > 0.5) == val_acc
                val_conf.append(conf_val.double().mean().item()) # conf_val.double().mean().detache().item()
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Val loss: {val_loss[-1]}, Val confidence: {val_conf[-1]}")

            # early stopping criteria. If validation loss doesn't change/improve for 
            # early_stopping_patience consecutive epoch, stop training early.
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                epoch_no_improv = 0
            else:
                epoch_no_improv += 1

            if epoch_no_improv >= early_stopping_patience:
                print(f"Early stop at epoch {epoch}, no improvement for {early_stopping_patience} epochs.")
                break

        model.train_loss = train_loss
        model.train_conf = train_conf
        model.val_loss = val_loss
        model.val_conf = val_conf
        
        return model
