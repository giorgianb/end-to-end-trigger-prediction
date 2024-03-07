"""
PyTorch dataset specifications.
"""

import torch.utils.data 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import torch
import dataclasses
from collections import defaultdict

def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate

    if name == 'gt-trkvec-masked':
        from . import gt_trkvec_masked

        train_dataset, valid_dataset, test_dataset = gt_trkvec_masked.get_datasets(**data_args)
        def collate_batch(batch):
            track_vector, trigger, n_tracks, is_trigger_track, momentums, energies, ip, origin_vertices, ptypes = [], [], [], [], [], [], [], [], []
            for batch_item in batch:
                track_vector.append(batch_item.track_vector)
                n_tracks.append(batch_item.n_tracks)
                is_trigger_track.append(batch_item.is_trigger_track)
                trigger.append(batch_item.trigger)
                momentums.append(batch_item.momentums)
                ip.append(batch_item.ip)
                energies.append(batch_item.energies)
                origin_vertices.append(batch_item.origin_vertices)
                ptypes.append(batch_item.ptypes)

            trigger = torch.tensor(trigger, dtype=torch.int64)
            ip = torch.stack(ip, dim=0)
            n_tracks = torch.tensor(n_tracks, dtype=torch.int64)
            track_vector = pad_sequence(track_vector, batch_first=True, padding_value=0)
            is_trigger_track = pad_sequence(is_trigger_track, batch_first=True, padding_value=0)
            momentums = pad_sequence(momentums, batch_first=True, padding_value=0)
            energies = pad_sequence(energies, batch_first=True, padding_value=0)
            origin_vertices = pad_sequence(origin_vertices, batch_first=True, padding_value=0)
            ptypes = pad_sequence(ptypes, batch_first=True, padding_value=0)

            return gt_trkvec_masked.BatchInfo(
                    track_vector=track_vector,
                    n_tracks=n_tracks,
                    trigger=trigger,
                    is_trigger_track=is_trigger_track,
                    momentums=momentums,
                    energies=energies,
                    ip=ip,
                    origin_vertices=origin_vertices,
                    ptypes=ptypes
            )

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader
    elif name == 'gt-hits-trkvec-masked':
        from . import gt_hits_trkvec_masked

        train_dataset, valid_dataset, test_dataset = gt_hits_trkvec_masked.get_datasets(**data_args)
        def collate_batch(batch):
            track_vector, trigger, n_tracks, is_trigger_track, momentums, energies, ip, origin_vertices = [], [], [], [], [], [], [], []
            for batch_item in batch:
                track_vector.append(batch_item.track_vector)
                n_tracks.append(batch_item.n_tracks)
                is_trigger_track.append(batch_item.is_trigger_track)
                trigger.append(batch_item.trigger)
                momentums.append(batch_item.momentums)
                ip.append(batch_item.ip)
                energies.append(batch_item.energies)
                origin_vertices.append(batch_item.origin_vertices)

            trigger = torch.tensor(trigger, dtype=torch.int64)
            ip = torch.stack(ip, dim=0)
            n_tracks = torch.tensor(n_tracks, dtype=torch.int64)
            track_vector = pad_sequence(track_vector, batch_first=True, padding_value=0)
            is_trigger_track = pad_sequence(is_trigger_track, batch_first=True, padding_value=0)
            momentums = pad_sequence(momentums, batch_first=True, padding_value=0)
            energies = pad_sequence(energies, batch_first=True, padding_value=0)
            origin_vertices = pad_sequence(origin_vertices, batch_first=True, padding_value=torch.nan)

            return gt_hits_trkvec_masked.BatchInfo(
                    track_vector=track_vector,
                    n_tracks=n_tracks,
                    trigger=trigger,
                    is_trigger_track=is_trigger_track,
                    momentums=momentums,
                    energies=energies,
                    ip=ip,
                    origin_vertices=origin_vertices
            )

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader

    elif name == 'pred-trkvec-masked':
        from . import pred_trkvec_masked

        train_dataset, valid_dataset, test_dataset = pred_trkvec_masked.get_datasets(**data_args)
        def collate_batch(batch):
            track_vector, trigger, n_tracks, is_trigger_track, momentums, energies, ip, origin_vertices, ptypes = [], [], [], [], [], [], [], [], []
            for batch_item in batch:
                track_vector.append(batch_item.track_vector)
                n_tracks.append(batch_item.n_tracks)
                is_trigger_track.append(batch_item.is_trigger_track)
                trigger.append(batch_item.trigger)
                momentums.append(batch_item.momentums)
                ip.append(batch_item.ip)
                energies.append(batch_item.energies)
                origin_vertices.append(batch_item.origin_vertices)
                ptypes.append(batch_item.ptypes)

            trigger = torch.tensor(trigger, dtype=torch.int64)
            ip = torch.stack(ip, dim=0)
            n_tracks = torch.tensor(n_tracks, dtype=torch.int64)
            track_vector = pad_sequence(track_vector, batch_first=True, padding_value=0)
            is_trigger_track = pad_sequence(is_trigger_track, batch_first=True, padding_value=0)
            momentums = pad_sequence(momentums, batch_first=True, padding_value=0)
            energies = pad_sequence(energies, batch_first=True, padding_value=0)
            origin_vertices = pad_sequence(origin_vertices, batch_first=True, padding_value=0)
            ptypes = pad_sequence(ptypes, batch_first=True, padding_value=0)

            return pred_trkvec_masked.BatchInfo(
                    track_vector=track_vector,
                    n_tracks=n_tracks,
                    trigger=trigger,
                    is_trigger_track=is_trigger_track,
                    momentums=momentums,
                    energies=energies,
                    ip=ip,
                    origin_vertices=origin_vertices,
                    ptypes=ptypes
            )

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader
    elif name == 'pred_trkvec_nomask':
        from. import pred_trkvec_masked
        return pred_trkvec_masked.get_data_loaders(name, batch_size, **data_args)
    
    elif name == 'gt-trkvec-nomasked-physics':
        from . import gt_trkvec_masked_physics
        return gt_trkvec_masked_physics.get_data_loaders(name, batch_size, **data_args)
    
    elif name == 'gt-trkvec-masked-physics':
        from . import gt_trkvec_masked_physics
        train_dataset, valid_dataset, test_dataset, train_data_n_nodes, valid_data_n_nodes, test_data_n_nodes = trkvec_ecml_masked_physics.get_datasets(**data_args)
        def collate_batch(batch):
            import torch
            track_list, trigger_list, lengths_list, momentum = [], [], [], []
            for (_track, _length, _trigger, _momentum) in batch:
                track_list.append(_track)
                lengths_list.append(_length)
                trigger_list.append(_trigger)
                momentum.append(_momentum)

            trigger_list = torch.tensor(trigger_list, dtype=torch.int64)
            length_list = torch.tensor(lengths_list, dtype=torch.int64)
            track_list = pad_sequence(track_list, batch_first=True, padding_value=0)
            momentum = pad_sequence(momentum, batch_first=True, padding_value=0)
            # print('############')

            return track_list, length_list, trigger_list, momentum

        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader
    elif name == 'pred-tracks':
        from . import pred_tracks

        train_dataset, valid_dataset, test_dataset = pred_tracks.get_datasets(**data_args)
        def collate_batch(batch):
            fields = defaultdict(list)
            track_level = defaultdict(bool)
            for batch_item in batch:
                batch_item = dataclasses.asdict(batch_item)
                for k, v in batch_item.items():
                    if v.ndim == 0:
                        fields[k].append(torch.tensor(v).unsqueeze(0))
                    else:
                        fields[k].append(torch.tensor(v))
                    track_level[k] = v.ndim == 2

            mask = None
            for k, v in fields.items():
                if track_level[k]:
                    if mask is None:
                        max_length = max(len(item) for item in v)
                        mask = torch.zeros((len(v), max_length), dtype=torch.float)
                        for i, item in enumerate(v):
                            mask[i, :len(item)] = True

                    fields[k] = pad_sequence(v, batch_first=True, padding_value=0)

                else:
                    try:
                        fields[k] = torch.cat(v, dim=0)
                    except Exception:
                        print(f'{v=}')
                        raise

            return pred_tracks.BatchInfo(**fields), mask

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader
    elif name == 'gt-hits-tracks':
        from . import gt_hit_tracks

        train_dataset, valid_dataset, test_dataset = gt_hit_tracks.get_datasets(**data_args)
        def collate_batch(batch):
            fields = defaultdict(list)
            track_level = defaultdict(bool)
            for batch_item in batch:
                batch_item = dataclasses.asdict(batch_item)
                for k, v in batch_item.items():
                    if v.ndim == 0:
                        fields[k].append(torch.tensor(v).unsqueeze(0))
                    else:
                        fields[k].append(torch.tensor(v))
                    track_level[k] = v.ndim == 2

            mask = None
            for k, v in fields.items():
                if track_level[k]:
                    if mask is None:
                        max_length = max(len(item) for item in v)
                        mask = torch.zeros((len(v), max_length), dtype=torch.float)
                        for i, item in enumerate(v):
                            mask[i, :len(item)] = True

                    fields[k] = pad_sequence(v, batch_first=True, padding_value=0)

                else:
                    try:
                        fields[k] = torch.cat(v, dim=0)
                    except Exception:
                        print(f'{v=}')
                        raise

            return gt_hit_tracks.BatchInfo(**fields), mask

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader
    elif name == 'tracks':
        from . import tracks
        DataLoader = torch.utils.data.DataLoader

        train_dataset, valid_dataset, test_dataset = tracks.get_datasets(**data_args)
        def collate_batch(batch):
            fields = defaultdict(list)
            track_level = defaultdict(bool)
            for batch_item in batch:
                batch_item = dataclasses.asdict(batch_item)
                for k, v in batch_item.items():
                    if v.ndim == 0:
                        fields[k].append(torch.tensor(v).unsqueeze(0))
                    else:
                        fields[k].append(torch.tensor(v))
                    track_level[k] = v.ndim == 2

            mask = None
            for k, v in fields.items():
                if track_level[k]:
                    if mask is None:
                        max_length = max(len(item) for item in v)
                        mask = torch.zeros((len(v), max_length), dtype=torch.float)
                        for i, item in enumerate(v):
                            mask[i, :len(item)] = True

                    try:
                        fields[k] = pad_sequence(v, batch_first=True, padding_value=0)
                    except Exception:
                        print(f'{k=} {[t.shape for t in v]=}')
                        raise

                else:
                    try:
                        fields[k] = torch.cat(v, dim=0)
                    except Exception:
                        print(f'{k=} {v=}')
                        raise

            return tracks.BatchInfo(**fields), mask

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader

    elif name == 'combined-tracks':
        from . import combined_tracks as tracks
        DataLoader = tracks.EpochDataLoader

        train_dataset, valid_dataset, test_dataset = tracks.get_datasets(**data_args)
        def collate_batch(batch):
            fields = defaultdict(list)
            track_level = defaultdict(bool)
            for batch_item in batch:
                batch_item = dataclasses.asdict(batch_item)
                for k, v in batch_item.items():
                    if v.ndim == 0:
                        fields[k].append(torch.tensor(v).unsqueeze(0))
                    else:
                        fields[k].append(torch.tensor(v))
                    track_level[k] = v.ndim == 2

            mask = None
            for k, v in fields.items():
                if track_level[k]:
                    if mask is None:
                        max_length = max(len(item) for item in v)
                        mask = torch.zeros((len(v), max_length), dtype=torch.float)
                        for i, item in enumerate(v):
                            mask[i, :len(item)] = True

                    try:
                        fields[k] = pad_sequence(v, batch_first=True, padding_value=0)
                    except Exception:
                        print(f'{k=} {[t.shape for t in v]=}')
                        raise

                else:
                    try:
                        fields[k] = torch.cat(v, dim=0)
                    except Exception:
                        print(f'{k=} {v=}')
                        raise

            return tracks.BatchInfo(**fields), mask

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader
    elif name == 'combined-tracks-full':
        from . import combined_tracks_full as tracks
        DataLoader = torch.utils.data.DataLoader

        train_dataset, valid_dataset, test_dataset = tracks.get_datasets(**data_args)
        def collate_batch(batch):
            fields = defaultdict(list)
            track_level = defaultdict(bool)
            for batch_item in batch:
                batch_item = dataclasses.asdict(batch_item)
                for k, v in batch_item.items():
                    if v.ndim == 0:
                        fields[k].append(torch.tensor(v).unsqueeze(0))
                    else:
                        fields[k].append(torch.tensor(v))
                    track_level[k] = v.ndim == 2

            mask = None
            for k, v in fields.items():
                if track_level[k]:
                    if mask is None:
                        max_length = max(len(item) for item in v)
                        mask = torch.zeros((len(v), max_length), dtype=torch.float)
                        for i, item in enumerate(v):
                            mask[i, :len(item)] = True

                    try:
                        fields[k] = pad_sequence(v, batch_first=True, padding_value=0)
                    except Exception:
                        print(f'{k=} {[t.shape for t in v]=}')
                        raise

                else:
                    try:
                        fields[k] = torch.cat(v, dim=0)
                    except Exception:
                        print(f'{k=} {v=}')
                        raise

            return tracks.BatchInfo(**fields), mask

            
        collate_fn = collate_batch

        loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
        train_sampler, valid_sampler, test_sampler = None, None, None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=n_ranks)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                    shuffle=(train_sampler is None), **loader_args)
        valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                            if valid_dataset is not None else None)
        test_data_loader = (DataLoader(test_dataset, sampler=test_sampler, **loader_args)
                            if test_dataset is not None else None)
        return train_data_loader, valid_data_loader, test_data_loader




    else:
        raise Exception('Dataset %s unknown' % name)

