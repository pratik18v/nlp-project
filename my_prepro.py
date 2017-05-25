#import sys
#sys.path.append('show-attend-and-tell/core')
from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
#from preprocess_overstock_data import *
from my_resize import *
import pandas as pd
import urllib2
from bs4 import BeautifulSoup

min_length = 4

############################################ Our Functions #####################################################
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def read_data_file(fp):

    hnames = ['title', 'links', 'raw_desc']
    data = pd.read_csv(fp, error_bad_lines=False, delimiter=',', header=None, names=hnames)
    #print ('Number of records before deleting rows: %s'%len(data))
    return data

'''
def get_caption(raw_sentence):
    """
    This function fetches the caption after cleaning the unnecessary tags
    """

    caption = ''
    if not raw_sentence.startswith("http"):
        try:
            print raw_sentence
            print '##################################################'
            raw_sentence = raw_sentence.split('content=')[1] #might need to remove quotes
            raw_sentence = raw_sentence.split('>')[0].strip()
        except:
            try:
                print raw_sentence
                print '##################################################'
                raw_sentence = raw_sentence.split('<p>')[1]
                raw_sentence = raw_sentence.split('</p>')[0].strip()
            except:
                #print (raw_sentence)
                print 'skipped'

        caption = raw_sentence.replace('"','')
        caption = caption.split('.')
        print caption
        print '##################################################'
        print '##################################################'
    return caption
'''

def get_caption(raw_sentence):
    """
    This function fetches the caption after cleaning the unnecessary tags
    """
    flag = False
    captions = []
    if raw_sentence.startswith("<span"):
        data = BeautifulSoup(raw_sentence, 'html.parser')
        caption = data.find('span').attrs['content']
        #raw_sentence = raw_sentence.split('content=')[1] #might need to remove quotes
        #raw_sentence = raw_sentence.split('>')[0].strip()
        flag = True
        #print raw_sentence
        #print '##################################################'
    elif raw_sentence.startswith("<div"):
        data = BeautifulSoup(raw_sentence, 'html.parser')
        caption = data.find('p').text
        #raw_sentence = raw_sentence.split('<p>')[1]
        #raw_sentence = raw_sentence.split('</p>')[0].strip()
        flag = True
        #print raw_sentence
        #print '##################################################'

    if flag:
        captions = caption.split('. ')
        captions = [strip_non_ascii(c) for c in captions]
        #print captions
        #print '##################################################'
        #print '##################################################'

    return captions


def process_caption_data(caption_data, max_length=15):
    '''
    This function will clean the captions and delete indices greater than or smaller than the given limit.
    '''
    #with open("./download_failed.txt", "r") as f:
    #    failed_idx = f.readlines()

    #failed_idx = [int(w.strip()) for w in failed_idx if int(w.strip())<len(caption_data)-1]
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        #print caption
        #print '###############################################'
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split()) <= min_length or len(caption.split()) > max_length:
            del_idx.append(i)

    #del_idx.extend(failed_idx)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)

    total_rows = len(caption_data)
#    to_be_dropped_rows = total_rows % 100
#    last_idx = total_rows - to_be_dropped_rows
#    caption_data = caption_data[:last_idx]

    print ('Number of records after deleting rows: %s'%len(caption_data))
    return caption_data

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

    caption_data = process_caption_data(caption_data, max_length)
    print ('Finish building captions.')
    return caption_data

def get_caption_data(fp, max_length=15):
    '''
    This function will read the overstock data and get the required columns.
    '''
    data = read_data_file('./data/title_image_links_docs.csv')#locally path might be different
    idx_file = open('image_id_rel.txt', 'r')
    idx_lines = idx_file.read().split('\n')[1:-1]
    idx_in_csv = []
    idx_in_name = []
    for il in idx_lines:
        #print il
        a, b = il.split(' ')
        idx_in_csv.append(int(a))
        idx_in_name.append(int(b))

    #this line is just for testiing, remove once code is properly tested on a smaller batch
    #data = data[:idx_in_name[-1]]

    image_ids = []
    file_names = []
    captions = []

    for idx, row in data.iterrows():
        if idx in idx_in_csv:
            if idx % 100 == 0:
                print (idx)

            raw_sentence = row['raw_desc']
            caption = get_caption(raw_sentence)
            #print caption
            #print '#############################################################'
            for c in caption:
                #print c
                image_ids.append(idx_in_name[idx_in_csv.index(idx)])
                f_name = fp +'%s.jpg'%(idx_in_name[idx_in_csv.index(idx)])
                file_names.append(f_name)
                captions.append(c)
            #print '######################################################################'
    caption_data = build_caption_data(image_ids, file_names, captions, max_length)
    return caption_data



#############################################################################################################



def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
#    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
#    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
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


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 15 #15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'

    #path to resized images
    i_fp = './image/2014_resized/'
    #n_images = 67691
    #building dataset
    print 'Start processing caption data'
    train_dataset = get_caption_data(i_fp, max_length)
    print 'Finished processing caption data'

    #train, val, and test --> 70, 15, and 15
    train_cutoff = int(0.70 * len(train_dataset))
    val_cutoff = int(0.85 * len(train_dataset))

    #path to data directory
    d_fp = './data'
    if not os.path.exists(d_fp+'/train'):
        os.makedirs(d_fp+'/train')
    if not os.path.exists(d_fp+'/val'):
        os.makedirs(d_fp+'/val')
    if not os.path.exists(d_fp+'/test'):
        os.makedirs(d_fp+'/test')

    save_pickle(train_dataset[:train_cutoff], d_fp+'/train/train.annotations.pkl')
    save_pickle(train_dataset[train_cutoff:val_cutoff].reset_index(drop=True), d_fp+'/val/val.annotations.pkl')
    save_pickle(train_dataset[val_cutoff+1:].reset_index(drop=True), d_fp+'/test/test.annotations.pkl')

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
                print start, '-', end
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()
