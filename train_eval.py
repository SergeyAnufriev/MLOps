import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


criterion = torch.nn.CrossEntropyLoss()
'''params will be later added to dict, for now the focus is right implementation for lightening module'''


class LightningTemplateModel(pl.LightningModule):
    def __init__(self,model,split_train_,split_valid_,test_dataset,collate_batch,BATCH_SIZE,LR):
        super().__init__()
        ''' to do: make hparams dict to intialise this class'''
        self.my_model     = model

        self.split_train_  = split_train_  # pytorch train dataset class
        self.split_valid_  = split_valid_  # pytorch valid dataset class
        self.test_dataset  = test_dataset  # pytorch test dataset class
        self.collate_batch = collate_batch # function which converts list of tuples [( label, 'hello world'),...()] into batch


        self.BATCH_SIZE    = BATCH_SIZE    # batch size in train/val/test
        self.LR           = LR             # initial learning rate at epoch 0

        '''
        params dict will be decided in the next meeting
        if hparams.model_type == 'Network1':
            self.__my_model = build_model1()
        else:
            self.__my_model = build_model2()
        '''

    def loss_acc(self, batch):
        '''Input: data batch
        Output: loss and accuracy on one batch'''

        label, text, offsets = batch
        predicted_label = self.my_model(text, offsets)
        loss = criterion(predicted_label, label)
        total_acc = (predicted_label.argmax(1) == label).sum().item()
        total_count = label.size(0)

        return loss, total_acc / total_count

    def forward(self, x):
        '''# in lightning, forward defines the prediction/inference actions'''

        return self.my_model(x)

    def training_step(self, batch, batch_idx):
        '''calculate loss and accuracy on one batch'''

        '''Gradient clipping performed inside trainer class, no need to clip
            inside trainning_step 
            https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html'''

        loss, acc = self.loss_acc(batch)
        self.log('train_loss', loss)
        self.log('train_acc', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        '''log validation loss and accuracy on one batch'''

        loss, acc = self.loss_acc(batch)
        self.log('val_loss',loss)
        self.log('val_acc',acc)


    def test_step(self,batch, batch_idx):
        '''log test loss and accuracy on one batch'''

        loss, acc = self.loss_acc(batch)
        self.log('test_loss', loss)
        self.log('test_acc',acc)


    def configure_optimizers(self):
        '''define optimizers and LR schedulers,
        returns optimizer and scheduler separtly
        https://github.com/PyTorchLightning/pytorch-lightning/issues/3795'''

        optimizer = torch.optim.SGD(self.my_model.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

        return [optimizer],[scheduler]

    '''To do: Separate dataloaders by creating lightning data class
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html'''

    def train_dataloader(self):
        return DataLoader(self.split_train_, batch_size=self.BATCH_SIZE,
                              shuffle=False, collate_fn=self.collate_batch)

    def val_dataloader(self):
        return DataLoader(self.split_valid_, batch_size=self.BATCH_SIZE,
                              shuffle=False, collate_fn=self.collate_batch)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE,
                             shuffle=False, collate_fn=self.collate_batch)











