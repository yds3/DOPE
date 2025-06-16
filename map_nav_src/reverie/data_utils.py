import os
import json
import jsonlines
import h5py
import numpy as np
import re
import nltk
import math
from utils.data import angle_feature

def read_category_file(infile):
    category_mapping = {}
    category_list = []
    category_number = {}
    with open(infile, 'r',encoding='utf-8') as f:
        next(f) # pass the first line
        for line in f:
            line = line.strip('\n').split('\t')  
            source_name, target_category = line[1], line[-1]
            category_mapping[source_name] = target_category
            if target_category not in category_list:
                category_list.append(target_category)
        category_list.append('others')
        for i,cat in enumerate(category_list):
            category_number[cat] = i
    return category_mapping, category_number

def preprocess_name(name,cat_mapping,cat_number,lem):
    ''' preprocess the name of object
    '''
    name = re.sub(r'[^\w\s]',' ',str(name).lower().strip())
    name = lem.lemmatize(name) # convert the word into root word
    name = ''.join([i for i in name if not i.isdigit()]) # remove number
    if name in cat_mapping:
        name = cat_mapping[name]
    else:
        name = name.split(' ')[0]
        if name in cat_mapping:
            name = cat_mapping[name]
        else:
            name = 'others'
    number = cat_number[name]
    return name, number

class ObjectFeatureDB(object):
    def __init__(self, obj_ft_file, obj_feat_size):
        self.obj_feat_size = obj_feat_size
        self.obj_ft_file = obj_ft_file
        self._feature_store = {}

    def load_feature(self, scan, viewpoint, max_objects=None):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.obj_ft_file, 'r') as f:
                obj_attrs = {}
                if key in f:
                    obj_fts = f[key][...][:, :self.obj_feat_size].astype(np.float32) 
                    for attr_key, attr_value in f[key].attrs.items():
                        obj_attrs[attr_key] = attr_value
                else:
                    obj_fts = np.zeros((0, self.obj_feat_size), dtype=np.float32)
            self._feature_store[key] = (obj_fts, obj_attrs)

        if max_objects is not None:
            obj_fts = obj_fts[:max_objects]
            obj_attrs = {k: v[:max_objects] for k, v in obj_attrs.items()}
        return obj_fts, obj_attrs

    def get_object_feature(
        self, scan, viewpoint, base_heading, base_elevation, angle_feat_size,
        max_objects=None
    ):
        obj_fts, obj_attrs = self.load_feature(scan, viewpoint, max_objects=max_objects)
        obj_ang_fts = np.zeros((len(obj_fts), angle_feat_size), dtype=np.float32)
        obj_box_fts = np.zeros((len(obj_fts), 3), dtype=np.float32)
        obj_ids = []
        if len(obj_fts) > 0:
            for k, obj_ang in enumerate(obj_attrs['directions']):
                obj_ang_fts[k] = angle_feature(
                    obj_ang[0] - base_heading, obj_ang[1] - base_elevation, angle_feat_size
                )
                w, h = obj_attrs['sizes'][k]
                obj_box_fts[k, :2] = [h/480, w/640]
                obj_box_fts[k, 2] = obj_box_fts[k, 0] * obj_box_fts[k, 1]
            obj_ids = obj_attrs['obj_ids']
        return obj_fts, obj_ang_fts, obj_box_fts, obj_ids


def load_instr_datasets(anno_dir, dataset, splits, tokenizer):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, 'REVERIE_%s_enc.json' % split)
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, 'REVERIE_%s_enc_xlmr.json' % split)
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, word_picker=None):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)


            if splits[0] == 'test':
                new_item['instruction'] = instr
                if word_picker is not None:
                    objects, actions = word_picker.pick_action_object_words(instr)
                    new_item['objects'] = objects
                    new_item['actions'] = actions
                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                new_item['path_id'] = item['id']  + '_' + str(j)
                new_item['instr_id'] = item['id'] + '_' + str(j)
                new_item['objId'] = None
                del new_item['instructions']
                del new_item['instr_encodings']
            else:
                if 'path_id' not in item.keys():
                    item['path_id'] = item['instr_id']


                if 'objId' in item:
                    new_item['instr_id'] = '%s_%s_%d' % (str(item['path_id']), str(item['objId']), j)
                else:
                    new_item['path_id'] = item['id']
                    new_item['instr_id'] = '%s_%d' % (item['id'], j)
                    new_item['objId'] = None
                new_item['instruction'] = instr
                if word_picker is not None:
                    objects, actions = word_picker.pick_action_object_words(instr)
                    new_item['objects'] = objects
                    new_item['actions'] = actions
                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                del new_item['instructions']
                del new_item['instr_encodings']
            data.append(new_item)
    return data

def load_obj2vps(bbox_file):
    obj2vps = {}
    bbox_data = json.load(open(bbox_file))
    for scanvp, value in bbox_data.items():
        scan, vp = scanvp.split('_')
        # for all visible objects at that viewpoint
        for objid, objinfo in value.items():
            if objinfo['visible_pos']:
                # if such object not already in the dict
                obj2vps.setdefault(scan+'_'+objid, [])
                obj2vps[scan+'_'+objid].append(vp)
    return obj2vps
