import torch
import torch.nn as nn
from model.gin_model import GNN, TextGNN
from model.bert import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim


class GINSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature,

            gin_hidden_dim,
            gin_num_layers,

            gat_hidden_dim,
            gat_num_layers,
            heads,

            drop_ratio,
            projection_dim,
            lr,
            weight_decay,
            bert_pretrain,
            bert_hidden_dim,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature

        self.gin_hidden_dim = gin_hidden_dim
        self.gin_num_layers = gin_num_layers

        self.gat_hidden_dim = gat_hidden_dim
        self.gat_num_layers = gat_num_layers
        self.heads = heads
        
        self.drop_ratio = drop_ratio
        self.projection_dim = projection_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.bert_pretrain = bert_pretrain
        self.bert_hidden_dim = bert_hidden_dim

        self.graph_encoder = GNN(
            num_layer=self.gin_num_layers,
            emb_dim=self.gin_hidden_dim,
            gnn_type='gin',
            # virtual_node=True,
            # residual=False,
            drop_ratio=self.drop_ratio,
            JK='last',
        )
        ckpt = torch.load('/home/zhongcl/momu_v2/Pretrain/gin_pretrained/graphMVP.pth')
        missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(ckpt, strict=False)
        print(missing_keys)
        print(unexpected_keys)

        # Text Encoder
        self.text_encoder = TextGNN(
            num_layer=self.gat_num_layers,
            emb_dim=self.gat_hidden_dim,
            heads=self.heads,
            drop_ratio=self.drop_ratio,
            JK='last',
        )

        # Smiles Encoder
        if self.bert_pretrain:
            self.smiles_encoder = TextEncoder(pretrained=False)
        else:
            self.smiles_encoder = TextEncoder(pretrained=True)

        if self.bert_pretrain:
            print("bert load kvplm")
            ckpt = torch.load('/home/zhongcl/momu_v2/Pretrain/kvplm_pretrained/ckpt_KV_1.pt')
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[20:]: v for k, v in ckpt.items()}
            elif 'bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[5:]: v for k, v in ckpt.items()}
            else:
                pretrained_dict = {"main_model."+k[12:]: v for k, v in ckpt.items()}
            self.smiles_encoder.load_state_dict(pretrained_dict, strict=False)


        self.graph_proj_head = nn.Sequential(
          nn.Linear(self.gin_hidden_dim, self.gin_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.gin_hidden_dim, self.projection_dim)
        )

        self.text_proj_head = nn.Sequential(
          nn.Linear(self.gat_hidden_dim, self.gat_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.gat_hidden_dim, self.projection_dim)
        )

        self.smiles_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )
        

    def forward(self, features_graph, features_text):
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        graph, text, smiles_ids, smiles_attention_mask = batch

        graph_rep = self.graph_encoder(graph)
        graph_rep = self.graph_proj_head(graph_rep)

        text_rep = self.text_encoder(text)
        text_rep = self.text_proj_head(text_rep)

        smiles_rep = self.smiles_encoder(smiles_ids, smiles_attention_mask)
        smiles_rep = self.smiles_proj_head(smiles_rep)

        _, _, loss1 = self.forward(graph_rep, text_rep)
        _, _, loss2 = self.forward(graph_rep, smiles_rep)
        _, _, loss3 = self.forward(smiles_rep, text_rep)

        loss = (loss1 + loss2 + loss3) / 3.0
        self.log("train_loss", loss)
        return loss


    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("GINSimclr")
    #     # train mode
    #     parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    #     # GIN
    #     parser.add_argument('--gin_hidden_dim', type=int, default=300)
    #     parser.add_argument('--gin_num_layers', type=int, default=5)
    #     parser.add_argument('--gin_pooling', type=str, default='sum')
    #     #GAT
    #     parser.add_argument('--gat_hidden_dim', type=int, default=768)
    #     parser.add_argument('--gat_num_layers', type=int, default=5)
    #     parser.add_argument('--gat_pooling', type=str, default='sum')
    #     parser.add_argument('--heads', type=int, default=2)

    #     parser.add_argument('--drop_ratio', type=float, default=0.0)
       
    #     # optimization
    #     parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    #     parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
    #     return parent_parser

