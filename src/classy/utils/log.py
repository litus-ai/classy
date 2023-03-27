import logging

_loggers = {}


def get_project_logger(module_name: str) -> logging.Logger:
    return _loggers.setdefault(module_name, logging.getLogger(module_name))
