#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:09:01 2017

@author: leena
"""

import pandas as pd
import urllib2
import os


def read_data_file(fp):
    
    hnames = ['title', 'links', 'raw_desc']
    data = pd.read_csv(fp, error_bad_lines=False, delimiter=',', header=None, names=hnames)
    #print ('Number of records before deleting rows: %s'%len(data))
    return data

def download_images(fp, n_images=20):
    """
    This function will download the images.
    """
    download_failed_list = []
    data = read_data_file('../overstock_data/title_image_links_docs.csv')
    if not os.path.exists(fp): 
        os.makedirs(fp)
    img_idx = 0    
    for idx, row in data.iterrows():
            
        if img_idx == n_images:
            break
        link = row['links'].strip()
        try:
            f_name = fp+'/'+'%s.jpg'%(img_idx)#idx
            #result = urllib.urlretrieve(link, f_name)
            img = urllib2.urlopen(link)
            localFile = open(f_name, 'wb')
            localFile.write(img.read())
            localFile.close()
            img_idx += 1
        except IOError:
            print 'Download failed for: {}'.format(f_name) ##todo handle cases where image is not properly loaded
            download_failed_list.append(idx)
            continue
    with open("./download_failed.txt", 'w') as f:
        for s in download_failed_list:
            f.write(str(s) + '\n') 

def get_caption(raw_sentence):
    """
    This function fetches the caption after cleaning the unnecessary tags
    """
    
    caption = '' 
    if not raw_sentence.startswith("http"):
        try:
            raw_sentence = raw_sentence.split('content="')[1]
            raw_sentence = raw_sentence.split('"><div>')[0]
            caption = raw_sentence.strip()
        except:
            try:
                raw_sentence = raw_sentence.split('><p>')[1]
                raw_sentence = raw_sentence.split('</p></div>')[0]
                caption = raw_sentence.strip()
            except:
                print (raw_sentence)

    return caption
    
def process_caption_data(caption_data, max_length=15):
    '''
    This function will clean the captions and delete indices greater than or smaller than the given limit.
    '''
    with open("./download_failed.txt", "r") as f:
        failed_idx = f.readlines()
        
    failed_idx = [int(w.strip()) for w in failed_idx]
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) == 0 or len(caption.split(" ")) > max_length:
            del_idx.append(i) 
    del_idx.extend(failed_idx)        
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

def get_caption_data(fp, n_images=20, max_length=15):
    '''
    This function will read the overstock data and get the required columns.
    '''
    data = read_data_file('../overstock_data/title_image_links_docs.csv')#locally path might be different
    #this line is just for testiing, remove one code is properly tested on a smaller batch
    data = data[:n_images]
    
    image_ids = []
    file_names = []
    captions = []
    
    for idx, row in data.iterrows():

        if idx % 100 == 0:
            print (idx)
    
        raw_sentence = row['raw_desc']
        raw_sentence = raw_sentence.split('.')[0]#taing the first sentence as of now.
        caption = get_caption(raw_sentence)
        
        image_ids.append(idx)
        f_name = fp +'%s.jpg'%(idx)
        file_names.append(f_name)
        captions.append(caption)

    caption_data = build_caption_data(image_ids, file_names, captions, max_length)  
    return caption_data   
           
        
       
    

    