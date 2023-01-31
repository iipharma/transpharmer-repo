import os.path as op
import pandas as pd
import numpy as np
import torch
from model import GPT
from utils.lr import LR
from utils.optim import build_optimizer
from utils.seed import set_seed
from dataset import build_dataloader
from utils.io import load_config
from dataset import SmileDataset
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = config.DEVICE
        self.config = config.TRAIN
        self.lr = LR(self.config.LR.LEARNING_RATE)
        self.train_loader = build_dataloader(train_dataset, True, self.config.BATCH_SIZE, self.config.NUM_WORKERS)
        self.valid_loader = build_dataloader(test_dataset, False, self.config.BATCH_SIZE, self.config.NUM_WORKERS)
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)
        self.scaler = GradScaler()
        self.best_loss = float('inf')
        self.trained_tokens = 0 # counter used for learning rate decay

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.CKPT_PATH)
        torch.save(raw_model.state_dict(), self.config.CKPT_PATH)

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        for it, (x, y, p) in enumerate(self.train_loader):
            # place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)
            p = p.to(self.device)
            # forward the model
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):
                    logits, loss = self.model(x, y, p)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
            # backprop and update the parameters
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.OPTIM.GRAD_NORM_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # decay the learning rate based on our progress
            self.trained_tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
            self.lr.step(self.trained_tokens)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr.lr
            # report progress
            if it % 100 == 0:
                print(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {self.lr.lr:e}")
            # pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
        return float(np.mean(losses))

    def valid_epoch(self, epoch):
        self.model.eval()
        losses = []
        for x, y, p in self.valid_loader:
            # place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)
            p = p.to(self.device)
            # forward the model
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits, loss = self.model(x, y, p)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
        test_loss = float(np.mean(losses))
        logger.info("test loss: %f", test_loss)
        return test_loss

    def train(self):
        for epoch in range(self.config.MAX_EPOCHS):
            self.train_epoch(epoch)
            test_loss = self.valid_epoch(epoch)
            print(f"epoch {epoch+1} valid loss {test_loss:.5f}")
            # supports early stopping based on the test loss, or just save always if no test set is provided
            if test_loss < self.best_loss:
                if self.config.CKPT_PATH is not None:
                    self.best_loss = test_loss
                    print(f'Saving at epoch {epoch + 1}')
                    self.save_checkpoint()


if __name__ == '__main__':
    config = load_config('configs/train.yaml')
    train_df = pd.read_csv(config.TRAIN.TRAIN_SET)
    valid_df = pd.read_csv(config.TRAIN.VALID_SET)
    
    train_dataset = SmileDataset(config,
                                 dataframe=train_df,
                                 block_size=config.MODEL.MAX_LEN,
                                 num_props=config.MODEL.NUM_PROPS,
                                 aug_prob=0.5)
    valid_dataset = SmileDataset(config,
                                 dataframe=valid_df,
                                 block_size=config.MODEL.MAX_LEN,
                                 num_props=config.MODEL.NUM_PROPS,
                                 aug_prob=0.5)

    model = GPT(config)

    ckpt_path = config.TRAIN.CKPT_PATH
    if op.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    config.TRAIN.WARMUP_TOKENS = 0.1 * len(train_dataset) * config.MODEL.MAX_LEN
    config.TRAIN.FINAL_TOKENS = config.TRAIN.MAX_EPOCHS * len(train_dataset) * config.MODEL.MAX_LEN
    trainer = Trainer(model, \
                      train_dataset, \
                      valid_dataset, \
                      config)
    trainer.train()