import os
import pydicom
import numpy as np
import pandas as pd

def preprocess(raw_data_path, array2image=False, save_path='../../processed_data/'):
    os.makedirs(save_path, exist_ok=True)
    array_save_path = os.path.join(save_path, 'arrays')
    os.makedirs(array_save_path, exist_ok=True)


    data_info = []

    unique_num = 0
    for (root, dirs, files) in os.walk(raw_data_path):
        for file in files:
            dcm_path = os.path.join(root, files)
            if dcm_path.endswith('.dcm'):
                unique_num +=1
                # read file
                dcm = pydicom.read_file(dcm_path)
                array = dcm.pixel_array

                if array2image:
                    array = array - array.min()
                    array = array / array.max()
                    array = array * 255
                    array = array.astype('uint8')

                path = os.path.abspath(os.path.join(array_save_path, '{:06d}.npy'.format(unique_num)))
                np.save(path, array)

                # TODO: 석회화 점수 입력
                score = -1

                data_info.append([unique_num, array_save_path, dcm_path, score])
    df = pd.DataFrame(data_info)
    df.columns = ['unique_number', 'array_path', 'dcm_path', 'score']

    df.to_csv(os.path.join(save_path, 'data_info.csv'), index=False)



if __name__ == '__main__':
    raw_data_path = '../../data/'
    preprocess(raw_data_path, array2image=False)