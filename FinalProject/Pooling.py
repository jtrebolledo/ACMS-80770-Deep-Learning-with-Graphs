import torch


class GraphPooling:
    """
        Graph pooling layer
    """
    def __init__(self, TypePooling):
        self.TypePooling = TypePooling
        pass

    def __call__(self, H):
        # -- multi-set pooling operator
        # H:batch_size*n*3
        # sum along the sencond axis, return batch_size*3 

        if self.TypePooling == "Sum":
            Pool = H.sum(dim=1)
        if self.TypePooling == "Max":
            Pool = H.max(dim=1)[0]
        if self.TypePooling == "Mean":
            Pool = H.mean(dim=1)

        return Pool