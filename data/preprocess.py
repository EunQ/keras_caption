import os
import csv
import numpy as np
import pickle
import re

# Req. 3-1	이미지 경로 및 캡션 불러오기
def get_path_caption():
    with open('configure.pkl', 'rb') as f:
        configure = pickle.load(f)
    return configure['img_dir_path'], configure['caption_file_path']


# Req. 3-2	전체 데이터셋을 분리해 저장하기
def dataset_split_save(cap_path):
    #image_cap_data, image_cap_test. idx로 구분.
    image_cap_list = []
    rowidx = 0
    with open(cap_path, newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter='|', )
        for row in rdr:
            if len(row) == 1:
                row = row[0].split('|')
            if rowidx == 0:
                rowidx += 1
                continue

            row[0] = "./datasets/images/" + row[0]
            row[2] = re.sub(',+$', '', row[2])
            image_cap_list.append([row[0], row[2]])

    image_cap = np.array(image_cap_list)
    np.random.shuffle(image_cap)
    pivot = int(len(image_cap) * 0.8)
    print("pivot : ", pivot)
    image_cap_data = image_cap[:pivot].copy()
    print(image_cap_data.shape)
    #print(image_cap_data)
    image_cap_test = image_cap[pivot:].copy()
    print(image_cap_test.shape)
    image_cap_data_path = "./datasets/image_cap_data"
    image_cap_test_path = "./datasets/image_cap_test"
    #print(type(image_cap_data_path), type(image_cap_data))
    #print(type(image_cap_test_path), type(image_cap_test))

    #np.save(image_cap_data_path. image_cap_data)
    np.save(image_cap_data_path, image_cap_data)
    np.save(image_cap_test_path, image_cap_test)

    return image_cap_data_path, image_cap_test_path

def get_data_list(cap_path, start_tkn = "<start>", end_tkn = "<end>"):
    #image_cap_data, image_cap_test. idx로 구분.
    image_cap_list = []
    rowidx = 0
    image_path_list = []
    text_list = []
    with open(cap_path, newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter='|', )
        for row in rdr:
            if len(row) == 1:
                row = row[0].split('|')
            if rowidx == 0:
                rowidx += 1
                continue

            row[0] = "./datasets/images/" + row[0]
            row[2] = row[2].lower()
            row[2] = re.sub(r'[^a-z]', ' ', row[2])
            #row[2] = re.sub(',+$', '', row[2])
            #row[2] = re.sub('[.]','',row[2])
            row[2] = row[2].strip()
            row[2] = start_tkn + " " + row[2] +' '+ end_tkn
            image_cap_list.append([row[0], row[2]])
    image_cap = np.array(image_cap_list)
    np.random.shuffle(image_cap)
    return image_cap

def get_word_dict(data):
    #data[0][0] = img_path
    #data[0][1] = caption.
    word_idx_dict = {}
    cnt = 3
    word_idx_dict['<start>'] = 1;
    word_idx_dict['<end>']   = 2;
    idx_word_dict = {}
    idx_word_dict[1] = '<start>'
    idx_word_dict[2] = '<end>'
    max_len = 0
    for idx, d in enumerate(data):
        word_cnt = 0
        for word in d[1].split():
            word_cnt+=1
            max_len = max(max_len, word_cnt)
            if word not in word_idx_dict:
                word_idx_dict[word] = cnt
                idx_word_dict[cnt] = word
                cnt += 1
        
    return word_idx_dict, idx_word_dict, max_len

    
# Req. 3-3	저장된 데이터셋 불러오기
def get_data_file(path):
    data = np.load(path+".npy")
    return np.split(data,2, axis=1)



# Req. 3-4	데이터 샘플링
def sampling_data(images, captions):
    idx = np.random.randint(len(images))
    return images[idx], captions[idx]


