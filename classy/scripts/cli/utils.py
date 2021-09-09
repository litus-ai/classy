def get_device(device):
    if device == "gpu" or device == "cuda":
        return 0

    if device == "cpu":
        return -1

    try:
        return int(device)
    except ValueError:
        pass

    return device  # TODO: raise NotImplemented?
