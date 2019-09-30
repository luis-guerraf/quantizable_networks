import importlib
import os
import time
import random

import torch
from torchvision import datasets, transforms
import numpy as np

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.config import FLAGS
from utils.meters import ScalarMeter, flush_scalar_meters

if FLAGS.model == 'models.s_resnet':
    save_filename = "_" + FLAGS.dataset + "_" + FLAGS.model + str(FLAGS.depth) + "_"
else:
    save_filename = "_" + FLAGS.dataset + "_" + FLAGS.model + "_"
save_filename += '_'.join(str(e) for e in FLAGS.width_mult_list) + "__" + \
                 '_'.join(str(e) for e in FLAGS.bitactiv_list) + "__" \
                 '_'.join(str(e) for e in FLAGS.bitwidth_list) + ".pt"

img_size = {'cifar10': 32, 'cifar100': 32, 'tiny_imagenet': 64, 'imagenet': 224}
num_classes = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(num_classes[FLAGS.dataset])
    return model


def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms == 'tiny_imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_scale = 0.08
        jitter_param = 0.4
        lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms == 'cifar10':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms == 'cifar100':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k' or FLAGS.dataset == 'tiny_imagenet':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'cifar10':
        if not FLAGS.test_only:
            train_set = datasets.CIFAR10(root=FLAGS.dataset_dir,
                             train=True, transform=train_transforms,
                             target_transform=None, download=False)
        else:
            train_set = None
        val_set = datasets.CIFAR10(root=FLAGS.dataset_dir,
                                     train=False, transform=val_transforms,
                                     target_transform=None, download=False)

        test_set = None
    elif FLAGS.dataset == 'cifar100':
        if not FLAGS.test_only:
            train_set = datasets.CIFAR100(root=FLAGS.dataset_dir,
                                         train=True, transform=train_transforms,
                                         target_transform=None, download=False)
        else:
            train_set = None
        val_set = datasets.CIFAR100(root=FLAGS.dataset_dir,
                                   train=False, transform=val_transforms,
                                   target_transform=None, download=False)

        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset_dir))
    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set):
    """get data loader"""
    if FLAGS.data_loader == 'imagenet1k_basic' or FLAGS.data_loader == 'tiny_imagenet' or \
            FLAGS.data_loader == 'cifar10' or FLAGS.data_loader == 'cifar100':
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=FLAGS.batch_size, shuffle=True,
                pin_memory=True, num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        else:
            train_loader = None
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=FLAGS.batch_size, shuffle=False,
            pin_memory=True, num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    return train_loader, val_loader, test_loader


def get_lr_scheduler(optimizer):
    """get learning rate"""
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            else:
                lr_dict[i] = lr_dict[i-1] * FLAGS.exp_decaying_lr_gamma
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - i / FLAGS.num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    FLAGS.lr_scheduler))
    return lr_scheduler


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # weight decay only on normal conv and fc
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


def set_random_seed():
    """set random seed"""
    if hasattr(FLAGS, 'random_seed'):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_meters(phase):
    """util function for meters"""
    if getattr(FLAGS, 'slimmable_training', False):
        meters_all = {}
        for width_mult in FLAGS.width_mult_list:
            for bitwidth in FLAGS.bitwidth_list:
                for bitactiv in FLAGS.bitactiv_list:
                    meters = {}
                    meters['loss'] = ScalarMeter('{}_loss/{}\t{}\t{}'.format(
                        phase, str(width_mult), str(bitwidth), str(bitactiv)))
                    for k in FLAGS.topk:
                        meters['top{}_error'.format(k)] = ScalarMeter(
                            '{}_top{}_error/{}\t{}\t{}'.format(phase, k, str(width_mult), str(bitwidth), str(bitactiv)))
                    meters_all[str(width_mult)+str(bitwidth)+str(bitactiv)] = meters
        meters = meters_all
    else:
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss'.format(phase))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error'.format(phase, k))
    return meters


def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    if getattr(FLAGS, 'slimmable_training', False):
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            for bitwidth in sorted(FLAGS.bitwidth_list, reverse=True):
                for bitactiv in sorted(FLAGS.bitactiv_list, reverse=True):
                    model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    model.apply(lambda m: setattr(m, 'bitwidth', bitwidth))
                    model.apply(lambda m: setattr(m, 'bitactiv', bitactiv))
                    print('Model profiling with: {}x {}bits {}activ:'.format(width_mult, bitwidth, bitactiv))
                    verbose = (width_mult == max(FLAGS.width_mult_list)) and (bitwidth == max(FLAGS.bitwidth_list)) and \
                                (bitactiv == max(FLAGS.bitactiv_list))
                    model_profiling(
                        model, img_size[FLAGS.dataset], img_size[FLAGS.dataset],
                        verbose=getattr(FLAGS, 'model_profiling_verbose', verbose))
    else:
        model_profiling(
            model, img_size[FLAGS.dataset], img_size[FLAGS.dataset],
            verbose=getattr(FLAGS, 'model_profiling_verbose', True))


def forward_loss(model, criterion, input, target, meter):
    """forward model and return loss"""
    output = model(input)
    loss = torch.mean(criterion(output, target))
    meter['loss'].cache(
        loss.cpu().detach().numpy())
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in FLAGS.topk:
        correct_k = correct[:k].float().sum(0)
        error_list = list(1.-correct_k.cpu().detach().numpy())
        meter['top{}_error'.format(k)].cache_list(error_list)
    return loss


def run_one_epoch(
        epoch, loader, model, criterion, optimizer, meters, phase='train'):
    """run one epoch for train/val/test"""
    t_start = time.time()
    assert phase in ['train', 'val', 'test'], "phase not be in train/val/test."
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
    if train and FLAGS.lr_scheduler == 'linear_decaying':
        linear_decaying_per_step = (
            FLAGS.lr/FLAGS.num_epochs/len(loader.dataset)*FLAGS.batch_size)
    for batch_idx, (input, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        if train:
            if FLAGS.lr_scheduler == 'linear_decaying':
                for param_group in optimizer.param_groups:
                    param_group['lr'] -= linear_decaying_per_step
            optimizer.zero_grad()
            if getattr(FLAGS, 'slimmable_training', False):
                for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                    for bitwidth in sorted(FLAGS.bitwidth_list, reverse=True):
                        for bitactiv in sorted(FLAGS.bitactiv_list, reverse=True):
                            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                            model.apply(lambda m: setattr(m, 'bitwidth', bitwidth))
                            model.apply(lambda m: setattr(m, 'bitactiv', bitactiv))
                            loss = forward_loss(
                                model, criterion, input, target,
                                meters[str(width_mult)+str(bitwidth)+str(bitactiv)])
                            loss.backward()
            else:
                loss, _ = forward_loss(
                    model, criterion, input, target, meters)
                loss.backward()

            # Optimize real value weights
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data)

        else:
            if getattr(FLAGS, 'slimmable_training', False):
                for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                    for bitwidth in sorted(FLAGS.bitwidth_list, reverse=True):
                        for bitactiv in sorted(FLAGS.bitactiv_list, reverse=True):
                            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                            model.apply(lambda m: setattr(m, 'bitwidth', bitwidth))
                            model.apply(lambda m: setattr(m, 'bitactiv', bitactiv))
                            forward_loss(
                                model, criterion, input, target,
                                meters[str(width_mult)+str(bitwidth)+str(bitactiv)])
            else:
                forward_loss(model, criterion, input, target, meters)
    if getattr(FLAGS, 'slimmable_training', False):
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            for bitwidth in sorted(FLAGS.bitwidth_list, reverse=True):
                for bitactiv in sorted(FLAGS.bitactiv_list, reverse=True):
                    results = flush_scalar_meters(meters[str(width_mult)+str(bitwidth)+str(bitactiv)])
                    print('{:.1f}s\t{:7s}{:6s}{} {}\t{}/{}: '.format(
                        time.time() - t_start, phase, str(width_mult), str(bitwidth), str(bitactiv), epoch,
                        FLAGS.num_epochs) + ', '.join('{}: {:.3f}'.format(k, v)
                                                        for k, v in results.items()))
    else:
        results = flush_scalar_meters(meters)
        print('{:.1f}s\t{:7s}\t{}/{}: '.format(
            time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
              ', '.join('{}: {:.3f}'.format(k, v) for k, v in results.items()))
    return results


def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    # seed
    set_random_seed()

    # model
    model = get_model()
    model_wrapper = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if (hasattr(FLAGS, 'pretrained_model_remap_keys') and
                FLAGS.pretrained_model_remap_keys):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                print('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        print('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = get_optimizer(model_wrapper)
    # check resume training
    if os.path.exists(os.path.join(FLAGS.log_dir, 'latest_checkpoint' + save_filename)):
        checkpoint = torch.load(
            os.path.join(FLAGS.log_dir, 'latest_checkpoint' + save_filename))
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = get_lr_scheduler(optimizer)
        lr_scheduler.last_epoch = last_epoch
        best_val = checkpoint['best_val']
        train_meters, val_meters = checkpoint['meters']
        print('Loaded checkpoint {} at epoch {}.'.format(
            FLAGS.log_dir, last_epoch))
    else:
        lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = lr_scheduler.last_epoch
        best_val = 1.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        val_meters['best_val'] = ScalarMeter('best_val')
        # if start from scratch, print model and do profiling
        # print(model_wrapper)
        if FLAGS.profiling:
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set = dataset(
        train_transforms, val_transforms, test_transforms)
    train_loader, val_loader, test_loader = data_loader(
        train_set, val_set, test_set)

    if FLAGS.test_only and (test_loader is not None):
        print('Start testing.')
        test_meters = get_meters('test')
        with torch.no_grad():
            run_one_epoch(
                last_epoch, test_loader, model_wrapper, criterion, optimizer,
                test_meters, phase='test')
        return

    print('Start training.')
    for epoch in range(last_epoch+1, FLAGS.num_epochs):
        lr_scheduler.step()
        # train
        results = run_one_epoch(
            epoch, train_loader, model_wrapper, criterion, optimizer,
            train_meters, phase='train')

        # val
        val_meters['best_val'].cache(best_val)
        with torch.no_grad():
            results = run_one_epoch(
                epoch, val_loader, model_wrapper, criterion, optimizer,
                val_meters, phase='val')
        if results['top1_error'] < best_val:
            best_val = results['top1_error']
            torch.save(
                {
                    'model': model_wrapper.state_dict(),
                },
                os.path.join(FLAGS.log_dir, 'best_model' + save_filename))
            print('New best validation top1 error: {:.3f}'.format(best_val))

        # save latest checkpoint
        torch.save(
            {
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
                'best_val': best_val,
                'meters': (train_meters, val_meters),
            },
            os.path.join(FLAGS.log_dir, 'latest_checkpoint' + save_filename))
    return


def main():
    """train and eval model"""
    train_val_test()


if __name__ == "__main__":
    main()
