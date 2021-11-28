import os, shutil
'''
    Inference와 Training 프로세스에 모두 영향을 미치는 파라미터는 config에서 변경해주는것이 유리합니다.
    예를들어 모델의 구조나 세부 동작(feature extractor등)변경은 inference와 training모두 변경이 필요합니다.
    GT의 형태(클래스의 개수)도 마찬가지로 inference와 training 과정에 모두 영향을 미치게 됩니다.

    하지만 loss function, optimizer, bathc_size, max_epoch, learning rate 변경등은
    Training 과정에 영향을 주지만, inference 과정에는 별다른 영향을 주지 않습니다.
    이러한 파라미터는 argparse를 이용하여 입력을 주어도 괜찮습니다. (혹은 config안에 모든것을 저장하는 패턴도 가능합니다.)
'''

#* 추가적으로 meta data관리도 해주시면 좋습니다. 
__author__ = "Gil Dong Hong, Cheol Min Park, Dong Soo Kim"
__copyright__ = "Copyright 2021, The CAC Project"
__credits__ = ["Dong Soo Kim"]
                    
__license__ = "GPL"
'''
    오픈소스 라이센스 간단 정리 (https://en.wikipedia.org/wiki/Open-source_license)
    GPL : (General Public License) 프로그램 배포시 무조건 동일한 GPL로 배포해야함
        : GPL은 세부적으로 GPLv2, GPLv2+, GPlv3, GPLv3+, Affero GPLv3등으로 나눠지며 짐(제약강화)
    MIT : 가장 느슨한 형태 저작권 명시만 하면됨)
    BSD : (Berkeley Software Distribution) 아무런 제한없이 누구나 자신의 용도로 사용, 저작권 명시만 하면됨
    Beerware : 조건이 매우 낮은 라이센스, 맥주 사주는 제약조건만 있으면 됨
'''

__version__ = "Major.Minor.Rel" # e.g. 1.0.1
__maintainer__ = "You Name"
__email__ = "han324solo@gmail.com"
__status__ = "Production / Development"
 

# [1] model architecture
MODEL_NAME = 'CACNet'
PRETRAINED = None # training resume을 위해서는 file_path를 명시합니다.

FEATURE_EXTRACTOR = 'resnet18'
FEATURE_PRETRAINED = True
FEATURE_FREEZE = False

INPUT_RESOLUTION = 224

# [2] ground truth
NUM_CLASS = 5


# [3] log
'''
    추가적으로 weight(pth)파일과 tensorboard event파일등을 저장하는 경로도 설정해 줍니다.
'''
cfg_baseline = 'attention'
cfg_num_class = 'cls' + str(NUM_CLASS)
cfg_feature_extractor = 'feat-' + FEATURE_EXTRACTOR

cfg_feature_list = [cfg_baseline, cfg_num_class, cfg_feature_extractor]
VERSION = '.'.join(cfg_feature_list)

SAVE_ROOT_DIR = '../save/' + VERSION

# [4] training param (optional)
'''
    training에 사용되는 파라미터를 명시해 줄 수도 있습니다.
    다음 코드는 예제 코드이며, 실제 train.py에서는 argparse를 이용하겠습니다.
'''
LOSS = 'cross-entropy'
OPTIMIZER = 'adam'
BATCH_SIZE = 32
MAX_EPOCH = 100
INITIAL_LEARNING_RATE = 1e-4
MILE_STONE = [40,40]

# [5] config file 복사
'''
    학습별로 config file을 복사하여 저장합니다.
    학습이 종료되고(혹은 학습중에) 세부 세팅에 대해서 참고할 수 있습니다.
'''
# TODO refactor utils.SaveManager class
#os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
#cfg_src_file_path = os.path.abspath(__file__)
#cfg_dst_file_path = os.path.join(SAVE_ROOT_DIR, __file__)
#shutil.copy2(cfg_src_file_path, cfg_dst_file_path)