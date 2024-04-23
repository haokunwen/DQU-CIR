

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import os

class DQU_CIR(nn.Module):
    def __init__(self, hidden_dim=1024, dropout = 0.5):
        super().__init__()
        # self.clip, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained=os.path.join('../models/laionCLIP-ViT-H-14-laion2B-s32B-b79K', 'open_clip_pytorch_model.bin'))
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        
        self.combiner_fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                         nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.scaler_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid())

    def extract_img_fea(self, x):
        image_features = self.clip.encode_image(x)
        return image_features

    def extract_text_fea(self, txt):
        txt = self.tokenizer(txt).cuda()
        text_features = self.clip.encode_text(txt)
        return text_features

    def extract_query(self, textual_query, visual_query):
        textual_query = F.normalize(self.extract_text_fea(textual_query), p=2, dim=-1)
        visual_query = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)
        combined_feature = self.combiner_fc(torch.cat([textual_query, visual_query], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * textual_query + (1 - dynamic_scaler) * visual_query
        return F.normalize(query, p=2, dim=-1)
    

    def extract_target(self, target_img):
        target_img_fea = self.extract_img_fea(target_img)
        return F.normalize(target_img_fea, p=2, dim=-1)

    def compute_loss(self, textual_query, visual_query, target_img):

        query_feature = self.extract_query(textual_query, visual_query) 
        target_feature = self.extract_target(target_img)  

        loss = {}
        loss['ranking'] = self.ranking_nce_loss(query_feature, target_feature)                                                                                         
        return loss

    def ranking_nce_loss(self, query, target):
        x = torch.mm(query, target.t())
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        loss = F.cross_entropy(self.loss_weight * x, labels)
        return loss
    

