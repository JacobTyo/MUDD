import sys
import time
import os.path as osp
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights
)
from tools.default_config import (
    imagedata_kwargs, optimizer_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)


def main():
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    cfg.merge_from_file('config.yaml')
    set_random_seed(cfg.train.seed)
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for image-reid'.format(cfg.loss.name)
    )
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        margin=cfg.loss.triplet.margin,
        weight_t=cfg.loss.triplet.weight_t,
        weight_x=cfg.loss.triplet.weight_x,
        scheduler=scheduler,
        use_gpu=cfg.use_gpu,
        label_smooth=cfg.loss.softmax.label_smooth)

    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
