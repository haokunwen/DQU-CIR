"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import string 
import glob
import pickle
import pathlib
import random
import cv2

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def draw_text(img, point, text, drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    fontScale = 0.7
    thickness = 5
    text_thickness = 2
    bg_color = (255, 255, 255)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (255, 0, 0), text_thickness, 8)
    elif drawType == "simple":
        cv2.putText(img, '%d' % (text), point, fontFace, 0.5, (255, 0, 0))
    return img

def draw_text_line(img, point, text_line: str, drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    fontScale = 0.7
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split(", ")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, drawType)
    return img
    

class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, category, transform=None, split='val-split'):
        super().__init__()

        self.path = path
        self.category = category
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        self.split = split

        if not os.path.exists(os.path.join(self.path, '{}_train_data.json'.format(self.category))):
            self.train_data = []
            self.train_init_process()
            with open(os.path.join(self.path, '{}_train_data.json'.format(self.category)), 'w') as f:
                json.dump(self.train_data, f)
        else:
            with open(os.path.join(self.path, '{}_train_data.json'.format(self.category)), 'r') as f:
                self.train_data = json.load(f) 
        
        with open(os.path.join(self.caption_dir, 'image_captions_{}_train.json'.format(self.category)), 'r') as f:
            self.train_captions = json.load(f)

        with open(os.path.join(self.caption_dir, 'keywords_in_mods_{}.json'.format(self.category)), 'r') as f:
            self.key_words = json.load(f)
        
        self.test_queries, self.test_targets = self.get_test_data()

    def train_init_process(self):
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.category, 'train')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(self.category)), 'r') as f:
            correction_dict = json.load(f)
        for triplets in ref_captions:
            ref_id = triplets['candidate']
            tag_id = triplets['target']
            cap = self.concat_text(triplets['captions'], correction_dict)
            self.train_data.append({
                'target': self.category + '_' + tag_id,
                'candidate': self.category + '_' + ref_id,
                'captions': cap
            })

    def correct_text(self, text, correction_dict):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

        return text

    def concat_text(self, captions, correction_dict):
        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
        return text

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        caption = self.train_data[idx]
        mod_str = caption['captions']
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['target_img_data'], out['target_img_path'] = self.get_img(target, stage=0) # 0:train 1:test
        out['mod'] = {'str': mod_str}
        out['textual_query'] = self.train_captions[candidate.split('_')[1]] + ', but ' + mod_str
        out['visual_query'], out['source_img_path'] = self.get_written_img(candidate, target, stage=0)

        return out

    def get_img(self, image_name, stage=0):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[stage](img)
        return img, img_path
    
    def get_written_img(self, candidate, target, stage=0):
        img_path = os.path.join(self.image_dir, candidate.split('_')[0], candidate.split('_')[1] + ".jpg")
        key_word = self.key_words[candidate.split('_')[1] + '_' + target.split('_')[1]][-1]

        candidate_img = cv2.imread(img_path)
        candidate_img = cv2.resize(candidate_img, (512, 512))
        written_img = draw_text_line(candidate_img, (15, 15), key_word)

        written_img = PIL.Image.fromarray(cv2.cvtColor(written_img, cv2.COLOR_BGR2RGB))
        written_img = self.transform[stage](written_img)
        return written_img, img_path


    def get_test_data(self):
        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(self.category, 'val')), 'r') as f:
            images = json.load(f)
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.category, 'val')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(self.category)), 'r') as f:
            correction_dict = json.load(f)
        with open(os.path.join(self.caption_dir, 'image_captions_{}_val.json'.format(self.category)), 'r') as f:
            img_captions = json.load(f)


        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = self.concat_text(caption['captions'], correction_dict)
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['visual_query'], out['source_img_path'] = self.get_written_img(self.category + '_' + candidate, self.category + '_' + target, stage=1)
            out['source_img_id'] = images.index(candidate)
            out['textual_query'] = img_captions[candidate] + ', but ' + mod_str
            out['target_img_id'] = images.index(target)
            out['target_img_data'], out['target_img_path'] = self.get_img(self.category + '_' + target, stage=1)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)

        test_targets_id = []
        test_targets = []
        if self.split == 'val-split':
            for i in test_queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])
            
            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'], out['target_img_path'] = self.get_img(self.category + '_' + images[i], stage=1)      
                test_targets.append(out)
        elif self.split == 'original-split':
            for id, image_name in enumerate(images):
                test_targets_id.append(id)
                out = {}
                out['target_img_id'] = id
                out['target_img_data'], out['target_img_path'] = self.get_img(self.category + '_' + image_name, stage=1)      
                test_targets.append(out)
        return test_queries, test_targets



class Shoes(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.transform = transform
        self.path = path


        with open(os.path.join(self.path, 'relative_captions_shoes.json')) as f:
            self.all_triplets = json.loads(f.read())
        
        train_image_file = 'train_im_names.txt'
        eval_image_file = 'eval_im_names.txt'
        train_image_file = open(os.path.join(self.path, train_image_file), 'r')
        train_image_names = train_image_file.readlines()
        train_image_names = [train_image_name.strip('\n') for train_image_name in train_image_names]

        eval_image_file = open(os.path.join(self.path, eval_image_file), 'r')
        eval_image_names = eval_image_file.readlines()
        eval_image_names = [eval_image_name.strip('\n') for eval_image_name in eval_image_names]

        self.imgfolder = os.listdir(self.path)
        self.imgfolder = [self.imgfolder[i] for i in range(len(self.imgfolder)) if 'womens' in self.imgfolder[i]]
        self.imgimages_all = []
        for i in range(len(self.imgfolder)):
            path = os.path.join(self.path,self.imgfolder[i])
            imgfiles = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
            self.imgimages_all += imgfiles
        self.imgimages_raw = [os.path.basename(imgname) for imgname in self.imgimages_all]

        with open(os.path.join(self.path, 'correction_dict_{}.json'.format('shoes')), 'r') as f:
            self.correction_dict = json.load(f)

        self.train_relative_pairs = []
        self.eval_relative_pairs = []
        for triplets in self.all_triplets:
            if triplets['ReferenceImageName'] in train_image_names:
                source = self.imgimages_all[self.imgimages_raw.index(triplets['ReferenceImageName'])]
                target = self.imgimages_all[self.imgimages_raw.index(triplets['ImageName'])]
                mod = triplets['RelativeCaption']
                self.train_relative_pairs.append({
                    'source': source,
                    'target': target,
                    'mod': mod.strip(),
                    'source_name': triplets['ReferenceImageName'],
                    'target_name': triplets['ImageName']
                })
            elif triplets['ReferenceImageName'] in eval_image_names:
                source = self.imgimages_all[self.imgimages_raw.index(triplets['ReferenceImageName'])]
                target = self.imgimages_all[self.imgimages_raw.index(triplets['ImageName'])]
                mod = triplets['RelativeCaption']
                self.eval_relative_pairs.append({
                    'source': source,
                    'target': target,
                    'mod': mod.strip(),
                    'source_name': triplets['ReferenceImageName'],
                    'target_name': triplets['ImageName']
                })

        with open(os.path.join(self.path, 'image_captions_shoes.json'), 'r') as f:
            self.all_captions = json.load(f)

        with open(os.path.join(self.path, 'keywords_in_mods_shoes.json'), 'r') as f:
            self.key_words = json.load(f)

        #self.img_w_mod_path = '/wen_hao_kun/Gemini/key_extract/shoes/shoes-shu-zuoshang-{}'.format(self.color)

        self.test_queries = self.get_test_queries()
        self.test_targets = self.get_test_targets()


    def correct_text(self, text):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([self.correction_dict.get(word) if word in self.correction_dict else word for word in tokens])
        return text

    def __len__(self):
        return len(self.train_relative_pairs)

    def __getitem__(self, idx):

        caption = self.train_relative_pairs[idx]
        out = {}
        candidate_name = caption['source_name']
        target_name = caption['target_name']
        mod_str = self.correct_text(caption['mod'])
        out['source_img_data'] = self.get_img(caption['source'], 0)
        out['target_img_data'] = self.get_img(caption['target'], 0)
        out['mod'] = {'str': mod_str}

        out['textual_query'] = self.all_captions[candidate_name] + ', but ' + mod_str
        out['visual_query'] = self.get_written_img(caption['source'], candidate_name, target_name, 0)

        return out
    
    def get_img(self, img_path, stage=0):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[stage](img)
        return img
    
    def get_written_img(self, source_img_path, source_img_name, target_img_name, stage=0):
        key_word = self.key_words[source_img_name + '+' + target_img_name][-1]
        candidate_img = cv2.imread(source_img_path)
        candidate_img = cv2.resize(candidate_img, (512, 512))
        written_img = draw_text_line(candidate_img, (15, 15), key_word)
        written_img = PIL.Image.fromarray(cv2.cvtColor(written_img, cv2.COLOR_BGR2RGB))
        written_img = self.transform[stage](written_img)
        return written_img

    def get_test_queries(self):
        test_queries = []
        for idx in range(len(self.eval_relative_pairs)):
            caption = self.eval_relative_pairs[idx]
            mod_str = self.correct_text(caption['mod'])
            candidate = caption['source']
            target = caption['target']
            candidate_name = caption['source_name']
            target_name = caption['target_name']

            out = {}
            out['source_img_id'] = self.imgimages_all.index(candidate)
            out['source_img_data'] = self.get_img(candidate, 1)
            out['target_img_id'] = self.imgimages_all.index(target)
            out['target_img_data'] = self.get_img(target, 1)
            out['mod'] = {'str': mod_str}
            out['textual_query'] = self.all_captions[candidate_name] + ', but ' + mod_str
            out['visual_query'] = self.get_written_img(candidate, candidate_name, target_name, 1)
            
            test_queries.append(out)
        return test_queries
    
    def get_test_targets(self):
        text_file = open(os.path.join(self.path, 'eval_im_names.txt'),'r')
        imgnames = text_file.readlines()
        imgnames = [imgname.strip('\n') for imgname in imgnames] # img list
        test_target = []
        for i in imgnames:
            out = {}
            out['target_img_id'] = self.imgimages_raw.index(i)
            out['target_img_data'] = self.get_img(self.imgimages_all[self.imgimages_raw.index(i)], 1)
            test_target.append(out)
        return test_target
    

class CIRR(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path 
        self.caption_dir = self.path + 'captions/captions/'
        self.split_dir = self.path + 'captions/image_splits/'
        self.transform = transform

        # train data
        with open(os.path.join(self.caption_dir, "cap.rc2.train.json"), 'r') as f:
            self.cirr_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys()) 

        with open(os.path.join(self.path, 'image_captions_cirr_train.json'), 'r') as f:
            self.train_captions = json.load(f)

        with open(os.path.join(self.path, 'keywords_in_mods_cirr_train.json'), 'r') as f:
            self.key_words_train = json.load(f)

        # val data
        with open(os.path.join(self.path, 'image_captions_cirr_val.json'), 'r') as f:
            self.val_captions = json.load(f)
        with open(os.path.join(self.path, 'keywords_in_mods_cirr_val.json'), 'r') as f:
            self.key_words_val = json.load(f)
        self.val_queries, self.val_targets = self.get_val_queries()

        # test data
        with open(os.path.join(self.path, 'image_captions_cirr_test1.json'), 'r') as f:
            self.test1_captions = json.load(f)
        with open(os.path.join(self.path, 'keywords_in_mods_cirr_test1.json'), 'r') as f:
            self.key_words_test1 = json.load(f)

        self.test_name_list, self.test_img_data, self.test_queries = self.get_test_queries()


    def __len__(self):
        return len(self.cirr_data)

    def __getitem__(self, idx):
        caption = self.cirr_data[idx]
        reference_name = caption['reference']
        mod_str = caption['caption']
        target_name = caption['target_hard']
        
        out = {}
        out['source_img_data'] = self.get_img(os.path.join('images', pathlib.Path(self.train_image_path[reference_name])), 0)
        out['target_img_data'] = self.get_img(os.path.join('images', pathlib.Path(self.train_image_path[target_name])), 0)
        out['mod'] = {'str':mod_str}

        out['textual_query'] = self.train_captions[reference_name] + ', but' + mod_str
        key_word = self.key_words_train[reference_name + '+' + target_name][-1]
        out['visual_query'] = self.get_written_img(os.path.join(self.path, 'images', pathlib.Path(self.train_image_path[reference_name])), key_word, 0)

        return out
    
    def get_written_img(self, source_img_path, key_word, stage=0):
        candidate_img = cv2.imread(source_img_path)
        written_img = draw_text_line(candidate_img, (15, 15), key_word)
        written_img = PIL.Image.fromarray(cv2.cvtColor(written_img, cv2.COLOR_BGR2RGB))
        written_img = self.transform[stage](written_img)
        return written_img

    
    def get_img(self, img_path, stage=0):
        img_path = os.path.join(self.path, img_path)
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[stage](img)
        return img


    def get_val_queries(self):
        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = list(val_image_path.keys())

        test_queries = []
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['caption']
            reference_name = caption['reference']
            target_name = caption['target_hard']
            subset_names = caption['img_set']['members']
            subset_ids = [val_image_name.index(n) for n in subset_names]

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_img(os.path.join(pathlib.Path(val_image_path[reference_name])), 1)
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_img(os.path.join(pathlib.Path(val_image_path[target_name])), 1)
            out['mod'] = {'str':mod_str}
            out['subset_id'] = subset_ids

            key_word = self.key_words_val[reference_name + '+' + target_name][-1]
            out['textual_query'] = self.val_captions[reference_name] + ', but' + mod_str
            out['visual_query'] = self.get_written_img(os.path.join(self.path, pathlib.Path(val_image_path[reference_name])), key_word, 1)

            test_queries.append(out)

        test_targets = []
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(os.path.join(pathlib.Path(val_image_path[name])), 1)
            test_targets.append(out)

        return test_queries, test_targets

    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        for i in range(len(test_data)):
            out = {}
            caption = test_data[i]
            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_img(os.path.join(pathlib.Path(test_image_path[caption['reference']])), 1)
            out['reference_name'] = caption['reference']
            out['mod'] = caption['caption']
            out['subset'] = caption['img_set']['members']

            out['textual_query'] = self.test1_captions[caption['reference']] + ', but' + caption['caption']
            key_word = self.key_words_test1[str(caption['pairid']) + '+' + caption['reference']][-1]
            out['visual_query'] = self.get_written_img(os.path.join(self.path, pathlib.Path(test_image_path[caption['reference']])), key_word, 1)
            queries.append(out)

        image_name = []
        image_data = []
        for i in range(len(test_image_name)):
            name = test_image_name[i]
            data = self.get_img(os.path.join(pathlib.Path(test_image_path[name])), 1)
            image_name.append(name)
            image_data.append(data)
        return image_name, image_data, queries


class Fashion200k(torch.utils.data.Dataset):
    """Fashion200k dataset."""

    def __init__(self, path, split='train', transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []
        self.test_queries = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                                         '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ', filename)
            with open(label_path + '/' + filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Fashion200k:', len(self.imgs), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
            self.stage = 0
        else:
            self.generate_test_queries_()
            self.stage = 1

    def get_loader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_test_queries(self):
        return self.test_queries

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries += [{
                'target_word': target_word,
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'mod': {
                    'str': mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                #if not caption2id.has_key(c):
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                #if not parent2children_captions.has_key(p):
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.imgs)
   
    def __getitem__(self, idx):
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)
        out = {}
        out['source_img_id'] = idx
        out['source_img_data'] = self.get_img(idx)
        out['source_caption'] = self.imgs[idx]['captions'][0]
        out['target_img_id'] = target_idx
        out['target_img_data'] = self.get_img(target_idx)
        out['target_caption'] = self.imgs[target_idx]['captions'][0]
        out['mod'] = {'str': mod_str}
        out['textual_query'] = self.imgs[idx]['captions'][0] + ', but ' + mod_str
        out['visual_query'] = self.get_written_img(idx, target_word)

        return out
    
    def get_written_img(self, idx, key_word):
        img = cv2.imread(os.path.join(self.img_path + self.imgs[idx]['file_path']))
        img = cv2.putText(img.copy(), key_word, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
        img = self.transform[self.stage](PIL.Image.fromarray(img))
        return img 
    
    def get_img(self, idx):
        img_path = self.img_path + self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        img = self.transform[self.stage](img)
        return img
