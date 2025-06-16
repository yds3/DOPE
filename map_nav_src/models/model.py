import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vlnbert_init import get_vlnbert_models
# 这个类的实现提供了一个灵活的框架，可以根据不同的模式处理不同类型的数据，并执行相应的任务
class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)
        
        if mode == 'language':            
            txt_embeds, obj_embeds = self.vln_bert(mode, batch)
            return txt_embeds, obj_embeds

        elif mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if self.args.dataset == 'reverie' and 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
                
            pano_embeds, pano_masks,pano_obj_embeds,pano_obj_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks,pano_obj_embeds,pano_obj_masks


        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s'%mode)

# Critic 类定义了一个简单的神经网络，用于估计输入状态的值。
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
