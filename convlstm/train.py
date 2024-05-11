
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchkeras import KerasModel 


dl_train = ...
dl_val =...

net = ...

model = KerasModel(net,
                   loss_fn=nn.CrossEntropyLoss(),
                   metrics_dict = {"acc":Accuracy()},
                   optimizer = torch.optim.Adam(net.parameters(),lr = 0.01)  )

model.fit(
    train_data = dl_train,
    val_data= dl_val,
    epochs=5,
    patience=2,
    monitor="val_acc", 
    mode="max")