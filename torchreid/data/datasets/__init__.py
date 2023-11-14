from __future__ import print_function, absolute_import

from .dataset import Dataset, ImageDataset, Market1501, performancePhoto
import torch
from torchreid.data.sampler import build_train_sampler

__image_datasets = {
    'market1501': Market1501,
    'performancephoto': performancePhoto
}


def dataloader(sources, transform, mode, combineall, root, split_id, batch_size, workers, use_gpu, drop_last,
               k_tfm=1, dataset_url=None, sampler=None, num_instances=None, num_datasets=None, verbose=True,
               endpoint=None, username=None, password=None, database_name=None, aws_access_key_id=None,
               aws_secret_access_key=None, region_name=None, s3_bucket=None
               ):
    data_set = []
    for name in sources:
        data_set_ = init_image_dataset(
            name,
            transform=transform,
            k_tfm=k_tfm,
            mode=mode,
            combineall=combineall,
            verbose=verbose,
            root=root,
            split_id=split_id,
            dataset_url=dataset_url,
            endpoint=endpoint,
            username=username,
            password=password,
            database_name=database_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            s3_bucket=s3_bucket
        )
        data_set.append(data_set_)
    data_set = sum(data_set)

    if sampler:
        data_loader_sampler = build_train_sampler(
            data_set.train,
            sampler,
            batch_size=batch_size,
            num_instances=num_instances,
            num_datasets=num_datasets
        )
    else:
        data_loader_sampler = None
    data_loader = torch.utils.data.DataLoader(
        data_set,
        sampler=data_loader_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_gpu,
        drop_last=drop_last
    )
    return data_set, data_loader



def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name](**kwargs)



def register_image_dataset(name, dataset):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __image_datasets[name] = dataset


