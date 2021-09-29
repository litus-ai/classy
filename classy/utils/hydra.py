from typing import Callable

import omegaconf


def fix_paths(conf, check_fn: Callable[[str], bool], fix_fn: Callable[[str], str]):
    if type(conf) == list or type(conf) == omegaconf.listconfig.ListConfig:
        for i in range(len(conf)):
            conf[i] = fix_paths(conf[i], check_fn=check_fn, fix_fn=fix_fn)
        return conf
    elif type(conf) == dict or type(conf) == omegaconf.dictconfig.DictConfig:
        for k, v in conf.items():
            conf[k] = fix_paths(v, check_fn=check_fn, fix_fn=fix_fn)
        return conf
    elif type(conf) == str:
        if "/" in conf and check_fn(conf):
            return fix_fn(conf)
        else:
            return conf
    elif type(conf) in [float, int, bool]:
        return conf
    elif conf is None:
        return conf
    else:
        raise ValueError(f"Unexpected type {type(conf)}: {conf}")
