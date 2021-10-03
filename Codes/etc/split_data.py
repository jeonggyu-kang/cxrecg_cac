import pandas as pd


def split(data_info_path):
    #TODO: dataset split 어떻게 할지 정한 후에 코딩
    data_info = pd.read_csv(data_info_path)

    # select train

    # select val

    # select test

    return

if __name__ == '__main__':
    data_info_path = '../../processed_data/data_info.csv'
    split(data_info_path)
