"""
Python module for holding our PyTorch models.
"""

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name == 'agnn_original':
        from .agnn_original import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'agnn':
        from .agnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'mpnn':
        from .mpnn import GNN
        return GNN(**model_args)
    elif name == 'noise_agnn':
        from .noise_agnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'agnn_trigger':
        from .agnn_trigger import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'hetero_gnn':
        from .heterogeneous_gnn import HeteroGNNSegmentClassifier
        return HeteroGNNSegmentClassifier(**model_args)
    else:
        raise Exception('Model %s unknown' % name)
