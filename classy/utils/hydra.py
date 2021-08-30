import os

import omegaconf
import hydra


def fix(conf):
    if type(conf) == list or type(conf) == omegaconf.listconfig.ListConfig:
        for i in range(len(conf)):
            conf[i] = fix(conf[i])
        return conf
    elif type(conf) == dict or type(conf) == omegaconf.dictconfig.DictConfig:
        for k, v in conf.items():
            conf[k] = fix(v)
        return conf
    elif type(conf) == str:
        if "/" in conf and os.path.exists(hydra.utils.to_absolute_path(conf[: conf.rindex("/")])):
            return hydra.utils.to_absolute_path(conf)
        else:
            return conf
    elif type(conf) in [float, int, bool]:
        return conf
    elif conf is None:
        return conf
    else:
        raise ValueError(f"Unexpected type {type(conf)}: {conf}")
