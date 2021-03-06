#import sys
#sys.path.append('show-attend-and-tell/core')
from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle as pkl
import hickle
import os
import json
from nltk.corpus import stopwords

#from preprocess_overstock_data import *
from my_resize import *
import pandas as pd
import urllib2
from bs4 import BeautifulSoup
import nltk
import string

min_length = 4

############################################ Our Functions #####################################################
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def read_data_file(fp):

    hnames = ['title', 'links', 'raw_desc']
    data = pd.read_csv(fp, error_bad_lines=False, delimiter=',', header=None, names=hnames)
    #print ('Number of records before deleting rows: %s'%len(data))
    return data


def get_caption(raw_sentence):
    """
    This function fetches the caption after cleaning the unnecessary tags
    """
    flag = False
    captions = []
    if raw_sentence.startswith("<span"):
        data = BeautifulSoup(raw_sentence, 'html.parser')
        caption = data.find('span').attrs['content']
        flag = True
        #print raw_sentence
        #print '##################################################'
    elif raw_sentence.startswith("<div"):
        data = BeautifulSoup(raw_sentence, 'html.parser')
        caption = data.find('p').text
        flag = True
        #print raw_sentence
        #print '##################################################'

    if flag:
        captions = caption.split('. ')
        captions = [strip_non_ascii(c) for c in captions]
        #print captions
        #print '##################################################'

    return captions


def process_caption_data(captions, max_length=15):#caption_data, max_length=15
    '''
    This function will clean the captions and delete indices greater than or smaller than the given limit.
    '''
    for i, caption in enumerate(captions):#caption_data['caption']
        #print caption
        #print '###############################################'
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        #caption = " ".join(nltk.word_tokenize(caption))  # replace multiple spaces
        #caption_data.set_value(i, 'caption', caption.lower())
        captions[i] = caption

    return captions #caption_data

def build_caption_data(image_ids, file_names, captions, max_length):
    '''
    This function will create the caption data object.
    '''

    print ('Start building captions.')
    hnames = ['caption', 'file_name', 'image_id']

    caption_data = pd.DataFrame(columns=hnames)
    caption_data['image_id'] = image_ids
    caption_data['caption'] = captions
    caption_data['file_name'] = file_names

#    caption_data = process_caption_data(caption_data, max_length)
    print ('Finish building captions.')
    return caption_data

def get_caption_data(fp, max_length=15):
    '''
    This function will read the overstock data and get the required columns.
    '''
    data = read_data_file('./data/title_image_links_docs.csv')
    title_data = []
    desc_data = []
    f_names = []

    idx_file = open('image_id_rel.txt', 'r')
    idx_lines = idx_file.read().split('\n')[1:-1]
    idx_in_csv = []
    idx_in_name = []
    for il in idx_lines:
        #print il
        a, b = il.split(' ')
        idx_in_csv.append(int(a))
        idx_in_name.append(int(b))

    image_ids = []
    file_names = []
    captions = []

    for idx, row in data.iterrows():
        flag = False
        if idx in idx_in_csv:
            if idx % 200 == 0:
                print idx

            raw_sentence = row['raw_desc']
            caption = get_caption(raw_sentence)#get a list based on splitting "."
            #print caption
            #print '#############################################################'
            caption = process_caption_data(caption, max_length)
            temp_caption = ''
            for c in caption:
                #print c
                if len(c.split()) > min_length and len(c.split()) <= max_length:
                #if len(nltk.word_tokenize(c)) > min_length and len(nltk.word_tokenize(c)) <= max_length:
                    flag = True
                    image_ids.append(idx_in_name[idx_in_csv.index(idx)])
                    f_name = fp +'%s.jpg'%(idx_in_name[idx_in_csv.index(idx)])
                    file_names.append(f_name)
                    captions.append(c)
                    temp_caption += ' ' + c
            if flag:
                title_data.append(row['title'])
                desc_data.append(temp_caption)
                f_names.append(f_name)
            #print '######################################################################'
    caption_data = build_caption_data(image_ids, file_names, captions, max_length)

    title_data = pd.DataFrame({'file_name': f_names, 'title':title_data, 'raw_desc':desc_data})
    print ('Number of records: %s and %s'%(len(caption_data), len(title_data)))
    return caption_data, title_data

#############################################################################################################

def _build_vocab(annotations, threshold=3):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        #words = nltk.word_tokenize(caption)
        for w in words:
            counter[w] +=1

        if len(caption.split(' ')) > max_len:
            max_len = len(caption.split(' '))

        #if len(nltk.word_tokenize(caption)) > max_len:
        #    max_len = len(nltk.word_tokenize(caption))

    vocab = [word for word in counter if counter[word] >= threshold]
#    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        #words = nltk.word_tokenize(caption)
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
#    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs

def get_attribute_feats(title_data):

    n_dim = 512

    ###### title tokens
    title_tokens = []
    intersect_tokens = []
    for idx, row in title_data.iterrows():
        tok1 = row['title'].lower().split()
        tok2 = row['raw_desc'].lower().split()
        intersect_token = list(set(tok2).intersection(set(tok1)))
        intersect_tokens.append(intersect_token)
        title_tokens.extend(intersect_token)

    #title_data['title_tokens'] = intersect_tokens

    return intersect_tokens

def make_att_info(filename, word_to_idx):

    data = pkl.load(open(filename, 'rb'))

    #Pruning
    mod_data = []
    for dat in data:
        temp = []
        for d in dat:
            if d.encode('utf-8').lower() not in stopwords.words('english'):
                try:
                    int(d)
                except ValueError:
                    temp.append(d.encode('utf-8').lower())
        mod_data.append(temp)

    #Converting words to idxs
    title_data = []
    for md in mod_data:
        temp = []
        for d in md:
            try:
                temp.append(word_to_idx[d])
            except KeyError:
                continue
        title_data.append(temp)

    #Padding
    pad_length = 4
    final_data = []
    for td in title_data:
        if len(td) >= pad_length:
            final_data.append(td[:pad_length])
        else:
            temp = td
            while len(temp) < pad_length:
                temp.append(word_to_idx['<NULL>'])
            final_data.append(temp)

    return final_data


if __name__ == "__main__":

    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 15#15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'

    #path to resized images
    i_fp = './image/2014_resized/'
    #n_images = 67691
    #building dataset
    print 'Start processing caption data'
    train_dataset, title_data = get_caption_data(i_fp, max_length)
    print 'Finished processing caption data'


    print 'Generating attribute features'
    feats_list = get_attribute_feats(title_data)
    print 'Finished generating attribute features'

    #train, val, and test --> 70, 15, and 15
    train_cutoff = int(0.70 * len(train_dataset))
    val_cutoff = int(0.85 * len(train_dataset))

    #path to data directory
    d_fp = './new_data/'

    if not os.path.exists(d_fp+'/train'):
        os.makedirs(d_fp+'/train')
    if not os.path.exists(d_fp+'/val'):
        os.makedirs(d_fp+'/val')
    if not os.path.exists(d_fp+'/test'):
        os.makedirs(d_fp+'/test')

    tr_t = train_dataset[:train_cutoff]
    val_t = train_dataset[train_cutoff:val_cutoff].reset_index(drop=True)
    t_t = train_dataset[val_cutoff+1:].reset_index(drop=True)#chceck later

    u_tr_t = len(tr_t['file_name'].unique())
    u_val_t = len(val_t['file_name'].unique())
    u_t_t = len(t_t['file_name'].unique())

    total_len = u_tr_t + u_val_t + u_t_t

    assert total_len == len(title_data)

    save_pickle(tr_t, d_fp+'/train/train.annotations.pkl')
    save_pickle(val_t, d_fp+'/val/val.annotations.pkl')
    save_pickle(t_t, d_fp+'/test/test.annotations.pkl')

    #saving the attribute features
    save_pickle(feats_list[:u_tr_t], d_fp+'/train/train.att.tokens.pkl')
    save_pickle(feats_list[u_tr_t:u_tr_t+u_val_t], d_fp+'/val/val.att.tokens.pkl')
    save_pickle(feats_list[u_tr_t+u_val_t:], d_fp+'/test/test.att.tokens.pkl')

    ################# train, val, and test data saved #####################

    for split in ['train', 'val', 'test']:
        annotations = load_pickle(d_fp+'/%s/%s.annotations.pkl' % (split, split))

        if split == 'train':
            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, d_fp+'/%s/word_to_idx.pkl' % split)

        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, d_fp+'/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, d_fp+'/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, d_fp+'/%s/%s.image.idxs.pkl' % (split, split))

        att_filename = d_fp+'/%s/%s.att.tokens.pkl' % (split, split)
        att_info = make_att_info(att_filename, word_to_idx)
        save_pickle(att_info, d_fp+'/%s/%s.att.info.pkl' %(split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, d_fp+'/%s/%s.references.pkl' % (split, split))
        print "Finished building %s caption dataset" %split

    #extract conv5_3 feature vectors
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in ['train', 'val', 'test']:
            anno_path = d_fp+'/%s/%s.annotations.pkl' % (split, split)
            save_path = d_fp+'/%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = list(annotations['file_name'].unique())
            n_examples = len(image_path)

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                #print start, '-', end
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))
