from torch.utils.data.dataset import Dataset
import os
import os.path as osp
import glob
from PIL import Image
import random

# Aff-Wild2 dataset
# class_names_original = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']  #ABAW3的标注
# label_index = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}  #MELD的标注
class_mapping = [0, 6, 5, 2, 4, 3, 1, 7]



class AffwildDataset(Dataset): 
    """load aff-wild2 dataset"""
    def __init__(self, data_list_dir, file_folder, anno_folder, split='train', transform_ops=None, downsampling=1):
        super().__init__()
        self.transforms = transform_ops 
        self.file_folder = file_folder
        # class_names_original = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']  #ABAW3的标注

        # label_index = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}  #MELD的标注
        class_mapping = [0, 6, 5, 2, 4, 3, 1, 7]  
        # class_mapping = [0, 0, 2, 1, 0, 0, 0, ]  # x,x, 2 , 1, x, x,x refomulate the class mapping

        self.downsampling = downsampling   # TODO delete later: test data set downsampling to speed up training
        self.split = split
        # if is_train:
        print('load Aff-Wild2_train...')
        data_list_path = osp.join(data_list_dir, f'{split}_masked_img_path_list.txt')
        anno_folder = osp.join(anno_folder, f'{split}_set')

        self.data_list = []
        # print(f'affwild2 data list path: {data_list_path}') 
        # raise ValueError('Penny stops here!!!')
        if os.path.isfile(data_list_path):
            print(f'  - Loading data list form: {data_list_path}')
            with open(data_list_path, 'r') as infile:
                self.data_list = [t.split(' ')[:2] for t in infile]   
            # TODO delete later
            if self.downsampling > 1:
                len_selected = len(self.data_list)//self.downsampling
                random.shuffle(self.data_list)
                self.data_list = self.data_list[:len_selected]
                
            return
        print(f'  - Generating data list form: {anno_folder}')
        self.data_list = self.gen_list(file_folder, anno_folder, save_path=data_list_path, class_mapping=class_mapping)

        print(f'  - Total images: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)//self.downsampling

    def __getitem__(self, index):
        """load each image"""
        # for resnet: 
        im = Image.open(os.path.join(self.file_folder, self.data_list[index][0]))
        data = self.transforms(im)
    
        label = self.data_list[index][1].strip()  # note: when read from .txt file, label is str
        try:
            return data, int(label)
        except Exception as e:
            print(f'expect data is not None')

    @staticmethod
    def gen_list(file_folder, anno_folder,  save_path=None, class_mapping=None):
        """Generate list of data samples where each line contains image path and its label
            Input:
                file_folder: folder path of images (aligned)
                anno_folder: folder path of annotations, e.g., ./EXPR_Classification_Challenge/  /root/data/aff-wild2/Third ABAW Annotations
                class_mapping: list, class mapping for negative and coarse
                save_path: path of a txt file for saving list, default None
            Output:
                out_list: list of tuple contains relative file path and its label 
        """
        out_list = []
        for label_file in glob.glob(os.path.join(anno_folder, '*.txt')):   
            with open(label_file, 'r') as infile:
                print(f'----- Reading labels from: {os.path.basename(label_file)}')
                vid_name = os.path.basename(label_file)[0:-4]
                for idx, line in enumerate(infile):
                    if idx == 0:
                        classnames = line.split(',')
                    else:
                        label = int(line)  #3
                        if label == -1 or label == 7: # Remove faces with the emotions '-1' and 'other'.
                            continue
                        if class_mapping != None:
                            label = class_mapping[label]  # 
                        
                        image_name = f'{str(idx).zfill(5)}.jpg'
                        if os.path.isfile(os.path.join(file_folder, vid_name, image_name)):
                            out_list.append((os.path.join(vid_name, image_name), str(label))) # tuple
        if save_path is not None:
            with open(save_path, 'w') as ofile:
                for path, label in out_list:
                    ofile.write(f'{path} {label}\n')
            print(f'List saved to: {save_path}')

        return out_list


cwd = osp.abspath(osp.dirname(__file__))
def get_affwild2_dataset(split, transform_ops, downsampling=1):
    root_path = osp.join(cwd, '../../common/data/aff-wild2') # '/home/penny/pycharmprojects/common/data/aff-wild2'
    data_list_dir = osp.join(root_path, 'preprocessed_data')
    data_folder = osp.join(root_path, 'cropped_all/cropped_aligned')
    # data_folder = osp.join(root_path, 'openface_masked_cropped')
    anno_folder = osp.join(root_path, '5th_ABAW_Annotations/EXPR_Classification_Challenge')
    return AffwildDataset(data_list_dir, data_folder, anno_folder, split, transform_ops, downsampling)