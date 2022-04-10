import torch
import pytorch_lightning as pl
from model_gen import TextClassificationModel


criterion = torch.nn.CrossEntropyLoss()
'''params will be later added to dict, for now the focus is right implementation for lightening module'''
LR = 5  # learning rate


class LightningTemplateModel(pl.LightningModule):
    def __init__(self):
        # Switch model by hparams
        '''
        params dict will be decided in the next meeting
        if hparams.model_type == 'Network1':
            self.__my_model = build_model1()
        else:
            self.__my_model = build_model2()
        '''
        self.__my_model = TextClassificationModel()

    def forward(self, x):
        '''# in lightning, forward defines the prediction/inference actions'''

        return self.__mymodel(x)

    def training_step(self, batch, batch_idx):
        '''calculate loss one batch
        input: batch
        output: loss (no backward computation, just calculation)'''
        label, text, offsets = batch
        predicted_label = self.__mymodel(text, offsets)
        loss = criterion(predicted_label, label)

        '''remove += from documentation, since lightening will do averaging itself'''
        train_acc = (predicted_label.argmax(1) == label).sum().item()

        self.log('train_loss',loss)
        self.log('train_accuracy',train_acc)

        return loss

    '''Gradient clipping performed on training object later, no need to clip
    inside trainning_step 
    https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html'''

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss',loss)


    def test_step(self,batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return self.log('val_loss', loss)


    def configure_optimizers(self):
        '''define optimizers and LR schedulers,
        returns optimizer and scheduler separtly
        https://github.com/PyTorchLightning/pytorch-lightning/issues/3795'''

        optimizer = torch.optim.SGD(self.__mymodel.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

        return [optimizer],[scheduler]









