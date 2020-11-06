import warnings
import torch as t


class SNConfig():
    train_data_root = '/home/luokun/data/datasets/TDGYB/'
    test_data_root = ''
    load_model_path = ''

    batch_size = 4
    use_gpu = True
    multi_gpu = True
    print_freq = 20

    max_epoch = 100
    save_freq = 1
    lr = 0.001
    lr_decay = 0.9
    weight_decay = 0e-5
    momentum = 0.9

    resize = (768, 1024)
    crop_size = (768, 1024)
    neg_pos_ratio = 3

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        sncfg.device = t.device('cuda') if sncfg.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


sncfg = SNConfig()
