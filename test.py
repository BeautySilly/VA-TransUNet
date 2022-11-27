import os, random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Synapse_dataset
from utils import test_single_volume
from networks import SegUNet


def inference(model, test_save_path):
    db_test = Dataset(base_dir='./data/Synapse/test_vol_h5',
                      split="test_vol",
                      list_dir='./lists/lists_Synapse')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]

        
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=9, patch_size=[img_size, img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    for i in range(1, 9):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"
if __name__ == '__main__':
    deterministic = 1
    seed = 1234
    img_size = 224
    vit_name = "ViT-B_16"
    vit_patches_size = 16
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': './data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = "Synapse"
    num_classes = dataset_config[dataset_name]['num_classes']
    volume_path = dataset_config[dataset_name]['volume_path']
    Dataset = dataset_config[dataset_name]['Dataset']
    list_dir = dataset_config[dataset_name]['list_dir']
    z_spacing = dataset_config[dataset_name]['z_spacing']
    is_pretrain = True
    exp = 'TU_' + dataset_name + str(img_size)


    
    net = SegUNet(224).cuda()
    net.load_state_dict(torch.load("./save_model_pt/base_model/epoch_149.pth"))
    test_save_path = None
    inference(net, test_save_path)
