from PIL import Image
#from preprocess_overstock_data import *
import os
import urllib2
import pandas as pd

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
    data = read_data_file('./data/title_image_links_docs.csv')
    if not os.path.exists(fp):
        os.makedirs(fp)
    img_idx = 0
    idx_in_csv = []
    idx_in_name = []
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
            idx_in_csv.append(idx)
            idx_in_name.append(img_idx)
            img_idx += 1
        except IOError:
            print 'Download failed for: {}'.format(f_name) ##todo handle cases where image is not properly loaded
            download_failed_list.append(idx)
            continue
    with open("./download_failed.txt", 'w') as f:
        for s in download_failed_list:
            f.write(str(s) + '\n')
    f.close()
    with open('./image_id_rel.txt', 'w') as f1:
        f1.write('csv_index name_index\n')
        for i in range(len(idx_in_csv)):
            f1.write(str(idx_in_csv[i]) + ' ' + str(idx_in_name[i]) + '\n')

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main_resize_image():

    folder = './image/2014'
    resized_folder = './image/2014_resized/'

    #### download images
    #print 'Start downloading images.'
    #download_images(folder, 67691)
    #print ('Total downloaded images: %s.'%len(id_to_filename))

    #### resize images
    if not os.path.exists(resized_folder): #1 folder for train and test resize
        os.makedirs(resized_folder)

    print 'Start resizing images.'
    image_files = os.listdir(folder)
    num_images = len(image_files)
    for i, image_file in enumerate(image_files):
        try:
            with open(os.path.join(folder, image_file), 'r+b') as f:
                #with Image.open(f) as image:
                image = Image.open(f)
                image = resize_image(image)
                image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print 'Resized images: %d/%d' %(i, num_images)
        except IOError:
            print 'Image invalid: {}'.format(image_file)



if __name__ == "__main__":
    main_resize_image()
