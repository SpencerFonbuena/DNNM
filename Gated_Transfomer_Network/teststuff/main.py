# @Time    : 2021/07/21 19:00
# @Author  : SY.M
# @FileName: main.py

from data_process.create_dataset import Hourly_Dataset
from config.param_config import Config as c
from train_validate_test.train_and_validate import train_and_validate
from train_validate_test.train_and_test import train_and_test
from utils.run_with_pkl import run_with_pkl
from utils.check_csv import check_csvf
from module.for_TS.transformer import Transformer as TS_Transformer
from module.for_MTS.transformer import Transformer as MTS_Transformer

# 获取数据
all_data = Hourly_Dataset # this doesn't do shi right now
# 维度预览
print(f'Use archive:{c.archive.__name__}')
print(f'dataset name:{c.archive.file_name}')
print('dimension information:[BatchSize, feature_dim, time_step_len]')
print(
    f'dimension of train:[{all_data.train_len}, {None if c.archive.__name__ == "UCR" else all_data.feature_dim}, {all_data.time_step_len}]')
print(
    f'dimension of test:[{all_data.test_len}, {None if c.archive.__name__ == "UCR" else all_data.feature_dim}, {all_data.time_step_len}]')
print(f'class number:{all_data.class_num}\r\n===============================================================')

# select model
net_type = None
if c.archive.__name__ == 'UCR':
    net_type = TS_Transformer
elif c.archive.__name__ == 'MTS' or c.archive.__name__ == 'UEA':
    net_type = MTS_Transformer
assert net_type is not None, 'net is None!'

if __name__ == '__main__':
    # action list
    print("\033[0;34m%s\033[0m" % 'Input Your choice:\r\n'
                                  '1:train and validate to tune hyper-parameters\r\n'
                                  '2:train and test to evaluate the model\r\n'
                                  '3:test with saved model\r\n'
                                  '4:check result and parameters in csv file')
    todo = input('')
    assert todo in ['1', '2', '3', '4'], 'Input error!'
    if todo == '1':
        # k-fold cross-validation on the test set
        train_and_validate(net_class=net_type, all_data=all_data)
    elif todo == '2':
        # Test with the best hyperparameters on the validation set
        train_and_test(net_class=net_type, all_data=all_data)
    elif todo == '3':
        # Test with the trained model
        run_with_pkl()
    elif todo == '4':
        
    # View the contents of the .csv file
        check_csv()
