import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model.contrastive_gin import GINSimclr
# from torch_geometric.data import LightningDataset
from data_provider.pretrain_datamodule import GINPretrainDataModule
from data_provider.pretrain_dataset import GINPretrainDataset
from torch.nn.parallel import DistributedDataParallel as DDP


def main(args):
    pl.seed_everything(args.seed)

    # data
    # train_dataset = GINPretrainDataset(args.root, args.text_max_len, args.graph_aug1, args.graph_aug2)
    # dm = LightningDataset(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    dm = GINPretrainDataModule(batch_size=args.batch_size, num_workers=args.num_workers)


    # model
    model = GINSimclr(
        temperature=args.temperature,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_num_layers=args.gin_num_layers,

        gat_hidden_dim=args.gat_hidden_dim,
        gat_num_layers=args.gat_num_layers,
        heads=args.heads,

        bert_pretrain=args.bert_pretrain,
        bert_hidden_dim=args.bert_hidden_dim,

        drop_ratio=args.drop_ratio,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print('total params:', sum(p.numel() for p in model.parameters()))
    # model = DDP(model, find_unused_parameters=True)

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="/home/zhongcl/Molecular_checkpoints/gat_text/", every_n_epochs=1, save_top_k=-1))
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=True)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, strategy=strategy)

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train mode
    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    # GIN
    parser.add_argument('--gin_hidden_dim', type=int, default=300)
    parser.add_argument('--gin_num_layers', type=int, default=5)
    parser.add_argument('--gin_pooling', type=str, default='sum')
    # GAT
    parser.add_argument('--gat_hidden_dim', type=int, default=768)
    parser.add_argument('--gat_num_layers', type=int, default=5)
    parser.add_argument('--gat_pooling', type=str, default='sum')
    parser.add_argument('--heads', type=int, default=2)
    # Bert
    parser.add_argument('--bert_hidden_dim', type=int, default=768)
    parser.add_argument('--bert_pretrain', type=bool, default=True)

    parser.add_argument('--projection_dim', type=int, default=300)
    parser.add_argument('--drop_ratio', type=float, default=0.0)
    
    # optimization
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
    
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    

    # parser = Trainer.add_argparse_args(parser)
    # parser = GINSimclr.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule.add_argparse_args(parser)  # add data args
    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    main(args)
