import os, random
from utils import DiceLoss
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from datasets import Synapse_dataset, RandomGenerator
from networks import SegUNet



def trainer_synapse(model, snapshot_path):
    base_lr = 0.01
    num_classes = 9
    batch_size = 24
    db_train = Synapse_dataset(base_dir='./data/Synapse/train_npz',
                               list_dir='./lists/lists_Synapse', split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[224, 224])]))

    def worker_init_fn(worker_id):
        random.seed(1234 + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = 150
    max_iterations = max_epoch * len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                labs = label_batch[1, ...].unsqueeze(0) * 50
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            iterator.close()
            break

    return "Training Finished!"


if __name__ == '__main__':
    model = SegUNet(224)
    device = torch.device('cuda:0')
    model = model.to(device)
    model = model.cuda()
    snapshot_path = "./save_model_pt/"
    trainer_synapse(model, snapshot_path)
