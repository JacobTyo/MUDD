from __future__ import division, print_function, absolute_import
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import yaml
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)
import wandb
from torchreid import metrics
from torchreid.losses import CrossEntropyLoss, TripletLoss


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, use_gpu=True):
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.epoch = 0
        self.log_wandb = False

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, rank1, save_dir, is_best=False):
        names = self.get_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models[name].state_dict(),
                    'epoch': epoch + 1,
                    'rank1': rank1,
                    'optimizer': self._optims[name].state_dict(),
                    'scheduler': self._scheds[name].state_dict()
                },
                osp.join(save_dir, name),
                is_best=is_best
            )

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[-1]['lr']

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        wandb_config=None
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """

        # if wandb_config:
        #     self.log_wandb = True
        #     with open('config.yaml') as file:
        #         config = yaml.load(file, Loader=yaml.FullLoader)
        #     wandb.init(config=config)
        self.log_wandb = True

        if visrank and not test_only:
            print('visrank can be set to True only if test_only=True')
            visrank = False

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print('=> Start training')

        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )

            if (self.epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (self.epoch+1) % eval_freq == 0 \
               and (self.epoch + 1) != self.max_epoch:
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                self.save_model(self.epoch, rank1, save_dir)

        if self.max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            self.save_model(self.epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data)
            if self.log_wandb:
                wandb.log({'loss_summary': loss_summary,
                           'epoch': self.epoch,
                           'learning_rate': self.get_current_lr()})
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar(
                    'Train/lr', self.get_current_lr(), n_iter
                )

            end = time.time()

        self.update_lr()

    def forward_backward(self, data):
        raise NotImplementedError

    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            rank1, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )

            if self.writer is not None:
                self.writer.add_scalar(f'Test/{name}/rank1', rank1, self.epoch)
                self.writer.add_scalar(f'Test/{name}/mAP', mAP, self.epoch)

        return rank1

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs = data['img']
                pids = data['pid']
                camids = data['camid']
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.model(imgs)
                batch_time.update(time.time() - end)
                features = features.cpu()
                f_.append(features)
                pids_.extend(pids.tolist())
                camids_.extend(camids.tolist())
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        if self.log_wandb:
            tolog = {'Rank-{:<3}'.format(i + 1): cmc[i] for i in range(10)}
            tolog['mAP'] = mAP
            tolog['epoch'] = self.epoch
            tolog['learning_rate'] = self.get_current_lr()
            wandb.log(tolog)

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_{0}_{1}_{2}'.format(dataset_name, str(self.epoch),
                                                                         str(np.random.randint(1000000)))),
                topk=visrank_topk
            )

        return cmc[0], mAP

    def two_stepped_transfer_learning(
        self, epoch, fixbase_epoch, open_layers, model=None
    ):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(model, open_layers)
        else:
            open_all_layers(model)


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs = data['img']
        pids = data['pid']

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs = self.model(imgs)
        loss = self.criterion(outputs, pids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary


class ImageTripletEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_t=1,
            weight_x=1,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(ImageTripletEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs = data['img']
        pids = data['pid']

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.criterion_t(features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_x > 0:
            loss_x = self.criterion_x(outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()

        loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary