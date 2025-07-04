import json
import logging
import torch
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad


logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


# 这段代码定义了BERT模型的嵌入层，它将输入的词ID转换为嵌入向量，并结合位置信息和标记类型信息，最后通过层归一化和dropout处理，输出用于后续BERT层的嵌入表示。
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        # For action and object
        self.act_embedding = nn.Embedding(config.word_max_action,config.hidden_size) 
        self.obj_embedding = nn.Embedding(config.word_max_object,config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, act_txt=None, obj_txt=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # act_embeddings = self.act_embedding(act_txt)
        obj_embeddings = self.obj_embedding(obj_txt)


        return embeddings, None, obj_embeddings

#创建三个线性层 self.query、self.key 和 self.value，用于将输入的隐藏状态转换为查询（Q）、键（K）和值（V）。、
# 这个类实现了自注意力机制，它接收模型的隐藏状态，并计算每个位置的注意力分数，然后使用这些分数来加权值，生成上下文感知的表示。注意力分数可以用于分析模型关注输入序列的哪些部分。
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 这是一个加了两行
        self.config = config
        self.extra_dropout = nn.Dropout(0.2)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

# 这个类的作用是将自注意力层的输出进行进一步的处理，包括线性变换、dropout和层归一化，然后将结果与原始输入相加，得到最终的输出。这种设计允许模型在自注意力层和后续层之间保留信息，同时引入非线性和正则化，有助于提高模型的性能和泛化能力。
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# 这个类将自注意力机制和其后的输出处理整合在一起，形成了BERT模型中的一个完整的注意力模块。在BERT模型中，这样的模块会重复多次，构成模型的主体部分。每个模块都会对输入序列进行自注意力计算，然后通过输出层进行进一步的处理，最终输出序列的表示，这些表示可以用于下游的NLP任务
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

#这个类的作用是对自注意力模块的输出进行线性变换和非线性激活，为模型的下一层提供更丰富的表示。在BERT模型中，这种变换通常是为了增加模型的非线性能力，帮助模型捕捉更复杂的特征。激活函数的选择对模型的性能有重要影响，常见的激活函数包括GELU、ReLU、tanh等。
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# 这个类的作用是将中间层的输出进行进一步的处理，包括线性变换、dropout和层归一化，然后将结果与原始输入相加，得到最终的输出
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# 这个类代表了BERT模型中的一个完整的层，它将自注意力机制、中间变换层和输出层整合在一起，形成了一个可以重复使用的模块
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs
#这个类的作用是将多个 BertLayer 层组合成一个编码器，负责管理这些层的前向传播，并且根据配置输出每层的隐藏状态和注意力分数。在BERT模型中，编码器是模型的核心部分，它通过多层的自注意力机制和前馈网络处理输入序列，生成序列的高级表示。
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

# 这个类的作用是对BERT模型中 [CLS] 标记的隐藏状态进行线性变换和激活函数处理，生成一个用于分类任务的固定大小的向量。
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
# 这个类的作用是对BERT模型的隐藏状态进行进一步的变换，以适应下游任务的需求。
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
# 这个类的作用是将BERT模型的隐藏状态变换并映射到词汇表的维度，用于预测掩码词。在MLM任务中，模型需要预测输入序列中被随机掩盖的词。这个预测头通过引入一个变换层和一个线性映射层，使得模型能够输出每个词的概率分布，从而进行预测。偏置项的使用是为了在输出层引入额外的灵活性，允许模型为每个词汇表中的词学习一个独立的偏置
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
# 这个类的作用是提供一个简单的接口，用于获取BERT模型在MLM任务中的预测分数。
class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
# 这个类的关键特点是它能够处理来自不同源的隐藏状态和上下文，这使得它在处理需要结合不同信息源的任务时非常有用，例如问答系统或多模态学习任务。
class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

# BertXAttention 类通过结合 BertOutAttention 和 BertSelfOutput 组件，实现了一种能够处理内部隐藏状态和外部上下文信息的自注意力机制。
# 这种机制特别适用于需要结合额外信息源进行决策的任务，例如跨文档的信息检索、多模态学习任务等。
class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores
# GraphLXRTXLayer 类是一个多模态融合层，它结合了语言和视觉信息的处理。通过自注意力机制和前馈网络层，该类能够处理和融合来自不同模态的特征，适用于需要同时处理语言和视觉信息的应用，
# 这里改了BertXLayer 将注意力机制、中间层和输出层组合成一个完整的 Transformer 层，使模型能够逐步提取和组合输入特征信息。
class BertXLayer(nn.Module):
    def __init__(self, config):
        super(BertXLayer, self).__init__()
        self.attention = BertXAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        attention_output, attention_scores = self.attention(input_tensor, ctx_tensor, ctx_att_mask)
        attention_outputs = (attention_output, attention_scores)
        intermediate_output = self.intermediate(attention_outputs[0])
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

#这里改了BertXEncoder 通过多层 BertXLayer 实现一个深度的编码器结构，使模型能够逐层聚合输入和上下文信息。同时，用户可以选择输出每层的隐藏状态和注意力分数，以便进一步分析或使用。
class BertXEncoder(nn.Module):
    def __init__(self, config):
        super(BertXEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertXLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs, ctx, attention_mask):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(inputs, ctx, attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
# 跨模态编码器   用于实现一个多模态的融合层，结合了语言和视觉特征的自注意力机制以及跨模态注意力机制。这个类看起来是为了解决在多模态（如视觉和语言）的输入下，如何通过自注意力（self-attention）和跨注意力（cross-attention）对特征进行融合。
class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output
# LanguageEncoder 类是一个灵活的语言编码器，它可以根据配置选择是否更新预训练的BERT模型参数。这个类可以用于NLP任务中的语言模型部分，处理文本输入并生成编码后的表示。
class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

        self.act_obj_token_emb = nn.Embedding(1,config.hidden_size) # 0 for act, 1 for obj
        self.instr_actobj_encoder = BertOutAttention(config)
        self.instr_aug_linear = nn.Linear(config.hidden_size,1)
        self.instr_ori_linear = nn.Linear(config.hidden_size,1)
        self.instr_sigmoid = nn.Sigmoid()

        self.instr_fuse_layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, txt_embeds, txt_masks, act_embeds=None, act_masks=None, obj_embeds=None, obj_masks=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        if obj_embeds is not None: 
            obj_masks = extend_neg_masks(obj_masks) # Only use object tokens for REVERIE since the action tokens are not obvious in REVERIE.
            instr_aug_embeds = self.instr_actobj_encoder(txt_embeds, obj_embeds, obj_masks)[0]
            aug_linear_weight = self.instr_aug_linear(instr_aug_embeds)
            ori_linear_weight = self.instr_ori_linear(txt_embeds)
            aug_weight = self.instr_sigmoid(aug_linear_weight+ori_linear_weight)
            txt_embeds = torch.mul(aug_weight,instr_aug_embeds) + torch.mul((1-aug_weight),txt_embeds)  

        return txt_embeds

#    CrossmodalEncoder 类是一个跨模态编码器，它结合了语言和视觉信息的处理。通过一系列的 GraphLXRTXLayer 实例，该类能够处理和融合来自不同模态的特征，适用于需要同时处理语言和视觉信息的应用
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds
# ImageEmbeddings 类是一个多功能的图像特征处理模块，它能够处理不同类型的图像特征，并将其与其他相关信息（如位置和导航类型）结合起来，生成用于后续处理的特征表示。这个类的设计允许灵活地处理各种图像特征，并提供了一种机制来集成额外的编码器层，以进一步处理特征表示。
class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None
        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens

#LocalVPEncoder 类是一个专门处理局部视点信息的编码器，它结合了位置嵌入和跨模态编码器来生成视点相关的特征表示。这个类的设计允许灵活地处理视点信息，并提供了一种机制来融合语言信息，以生成更丰富的特征表示。
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds
# GlobalMapEncoder 类是一个专门处理全局地图信息的编码器，它结合了位置嵌入、步骤嵌入和跨模态编码器来生成全局地图相关的特征表示。这个类的设计允许灵活地处理全局地图信息，并提供了一种机制来融合语言信息，以生成更丰富的特征表示。
class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds
       
#     ClsPrediction 类是一个简单的分类预测头，它通过一个包含线性层、激活函数和层归一化的神经网络序列，将输入特征转换为最终的分类预测。这个类的设计适用于各种分类任务，特别是那些需要对输入特征进行复杂变换的任务。
class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = ImageEmbeddings(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)

        self.instr_actobj_encoder = BertOutAttention(config)
        self.instr_aug_linear = nn.Linear(config.hidden_size,1)
        self.instr_ori_linear = nn.Linear(config.hidden_size,1)
        self.instr_sigmoid = nn.Sigmoid()




        if config.glocal_fuse:
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        else:
            self.sap_fuse_linear = None
        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.config.hidden_size)

        self.knowledge_proj = nn.Linear(512, 768)
        self.crop_proj = nn.Linear(512, 768)
        self.instruction_proj = nn.Linear(768, 768)

        knowledge_config = copy.deepcopy(config)
        knowledge_config.num_x_layers = 2

        
        self.init_weights()
        
        if config.fix_lang_embedding or config.fix_local_branch:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding or config.fix_local_branch:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False
        if config.fix_local_branch:
            for k, v in self.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.local_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.og_head.named_parameters():
                v.requires_grad = False
    
    def forward_text(self, txt_ids, txt_masks, act_txt=None, act_txt_masks=None, obj_txt=None, obj_txt_masks=None):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        # print(f'333333333333333333333333333333333{obj_txt}')
        # print(f'333333333333333333333333333333333{act_txt}')

        txt_embeds, act_embeds, obj_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids,
                act_txt=act_txt, obj_txt=obj_txt
            )
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks, act_embeds, act_txt_masks, obj_embeds, obj_txt_masks)
        return txt_embeds, obj_embeds 

    def forward_panorama_per_step(
        self, view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens, instruction_fts, knowledge_fts,crop_fts, used_cand_ids, gmap_img_embeds, gmap_step_ids, gmap_pos_fts
    
    ):
        global CROP_SIZE
        batch_size = view_img_fts.size(0)

        knowledge_fts = self.knowledge_proj(knowledge_fts)
        knowledge_fts_shape = knowledge_fts.shape
        crop_fts = self.crop_proj(crop_fts)
        crop_fts_shape = crop_fts.shape
        instruction_fts_pure = self.instruction_proj(instruction_fts).permute(0,2,1).unsqueeze(1).repeat(1,36,1,1)    

          

        device = view_img_fts.device
        has_obj = obj_img_fts is not None

        view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(view_img_fts)
        )
        if has_obj:
            if self.img_embeddings.obj_linear is None:
                obj_img_embeds = self.img_embeddings.img_layer_norm(
                    self.img_embeddings.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.img_embeddings.obj_layer_norm(
                    self.img_embeddings.obj_linear(obj_img_fts)
                )
            # print(f"xianview_img_embeds shape2: {view_img_embeds.shape}[8, 37, 768]")
            # print(f"1111111111111111111obj_img_embeds shape: {obj_img_embeds.shape}[8, 13, 768]")

            img_embeds = []
            obj_embeds=[]
            for view_embed, obj_embed, view_len, obj_len in zip(
                view_img_embeds, obj_img_embeds, view_lens, obj_lens
            ):
                
                # print(f"obj_len shape: {obj_len}0,12,1")
                # print(f"view_len shape: {view_len}36")
                # print(f"11111111111obj_embed shape: {obj_embed.shape}[13, 768]")
                # print(f"view_embed shape: {view_embed.shape}[37, 768]")

                if obj_len > 0:

                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                    obj_embeds.append(obj_embed[:obj_len])

                else:
                    img_embeds.append(view_embed[:view_len])
                    obj_embeds.append(obj_embed[:obj_len])

            img_embeds = pad_tensors_wgrad(img_embeds)       
            obj_img_embeds = pad_tensors_wgrad(obj_embeds)
            pano_lens = view_lens + obj_lens

        else:
            img_embeds = view_img_embeds
            pano_lens = view_lens
        # print(f"11111111img_embeds shape2: {img_embeds.shape}")
        pano_embeds = img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
      
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        # print(f"1111111111111pano_embeds shape2: {pano_embeds.shape}[8, 49, 768]")
        # print(f"1111111111111111111obj_img_embeds shape: {obj_img_embeds.shape}[8, 13, 768]")
        # padding = (0, 0, 0, 49 - obj_img_embeds.size(1))  # pad second dimension
        # obj_img_embeds= F.pad(obj_img_embeds, padding, "constant", 0)
        # print(f"1111111111111111111obj_img_embeds shape: {obj_img_embeds.shape}[8, 49, 768]")

        max_len = pano_embeds.size(1)

        # 填充 obj_img_embeds 使其第二维度与 img_embeds 一致
        if obj_img_embeds.size(1) < max_len:
            padding = (0, 0, 0, max_len - obj_img_embeds.size(1))
            obj_img_embeds = F.pad(obj_img_embeds, padding, "constant", 0) 
        # print(f"1111111111111111111obj_img_embeds shape: {obj_img_embeds.shape}")

        
        pano_obj_embeds = obj_img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        pano_obj_embeds = self.img_embeddings.layer_norm(pano_obj_embeds)
        pano_obj_embeds = self.img_embeddings.dropout(pano_obj_embeds)



        pano_masks = gen_seq_masks(pano_lens)
        pano_obj_masks = gen_seq_masks(obj_lens)
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
            # pano_obj_embeds = self.img_embeddings.pano_encoder(
            #     pano_obj_embeds, src_key_padding_mask=pano_obj_masks.logical_not()
            # )


        return pano_embeds, pano_masks,pano_obj_embeds,pano_obj_masks
# 这个 forward_navigation_per_step 方法的主要功能是处理基于文本和视觉信息的导航任务，其中包含全局和局部分支的计算，融合全局与局部信息，并最终产生用于导航决策的 logits
    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,obj_embeds,obj_masks,vp_img_obj_embeds
    ):
        # print(f"vp_img_embeds shape: {vp_img_embeds.shape}  vp_img_obj_embeds{vp_img_obj_embeds.shape}")
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
       
        # local branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)

        vp_obj_embeds = vp_img_obj_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_obj_embeds = self.local_encoder.encoder(obj_embeds, obj_masks, vp_obj_embeds, vp_obj_masks)
        # 强调 vp_embeds 的权重
        # print(f"vp_obj_embeds{vp_obj_embeds.shape}")
        # vp_embeds= 0.8 * vp_embeds + 0.2 * vp_obj_embeds  # 重点强调 vp_embeds
        vp_obj_masks_copy=vp_obj_masks
        if vp_obj_embeds is not None: 
            vp_obj_masks = extend_neg_masks(vp_obj_masks)
            # print(f"111111 vp_obj_masks shape[6, 1, 1, 50]: {vp_obj_masks.shape}")
            instr_aug_embeds = self.instr_actobj_encoder(vp_embeds, vp_obj_embeds, vp_obj_masks)[0]
            aug_linear_weight = self.instr_aug_linear(instr_aug_embeds)
            ori_linear_weight = self.instr_ori_linear(vp_embeds)
            aug_weight = self.instr_sigmoid(aug_linear_weight+ori_linear_weight)
            vp_embeds = torch.mul(aug_weight,instr_aug_embeds) + torch.mul((1-aug_weight),vp_embeds)  


            
        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        # print(fuse_weights)

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        # print('global', torch.softmax(global_logits, 1)[0], global_logits[0])

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))
        # print('local', torch.softmax(local_logits, 1)[0], local_logits[0])

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
        # print('fused', torch.softmax(fused_logits, 1)[0], fused_logits[0])
        vp_obj_masks=vp_obj_masks_copy
        # object grounding logits
        # vp_obj_masks = vp_obj_masks.view(vp_obj_masks.size(0), -1)
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
        }
        return outs
    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked





    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds, obj_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'], 
                batch['act_txt_ids'], batch['act_txt_masks'], batch['obj_txt_ids'], batch['obj_txt_masks'])
            return txt_embeds, obj_embeds 

        elif mode == 'panorama':
            pano_embeds, pano_masks,pano_obj_embeds,pano_obj_masks= self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks,pano_obj_embeds,pano_obj_masks

        elif mode == 'navigation':
            
            return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
                batch['obj_embeds'],batch['obj_masks'],batch['vp_img_obj_embeds']
            )

                  