def count_params(model):
    """
    Count number of trainable parameters in PyTorch model
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params