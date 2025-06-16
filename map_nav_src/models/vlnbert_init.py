import torch
from transformers import AutoModel

# 这个函数的目的是根据不同的参数来动态选择并加载不同的预训练分词器。这样的设计使得代码更加灵活，可以根据不同的任务需求来选择合适的分词器。你正在处理多语言数据，你可能会选择使用 xlm-roberta-base 分词器；如果你主要处理的是英文数据，那么 bert-base-uncased 分词器可能更合适。
def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer
# 这个函数的目的是将预训练的BERT模型的权重适配到一个自定义的视觉语言模型中，并根据提供的参数配置模型
def get_vlnbert_models(args, config=None):  
    from transformers import PretrainedConfig
    if args.model == 'r2r':
        from models.vilmodel import GlocalTextPathNavCMT
    elif args.model == 'reverie':
        from models.vilmodel_reverie import GlocalTextPathNavCMT
    print("BERT checkpoint file path:", args.bert_ckpt_file)
    model_name_or_path = args.bert_ckpt_file
    print()
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v
            
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False

    vis_config.max_instr_len = args.max_instr_len
    vis_config.word_max_action = args.word_max_action
    vis_config.word_max_object = args.word_max_object
        
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model
