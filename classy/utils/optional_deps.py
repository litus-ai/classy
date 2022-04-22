import importlib
from typing import Optional


def requires(library, extra_required: Optional[str] = None):
    def closure(decorated_arg):
        def inner(*args, **kwargs):
            try:
                importlib.import_module(library)
            except ModuleNotFoundError as e:
                error_message = f"ModuleNotFoundError: {library} not found."
                if extra_required is not None:
                    error_message += f" It seems you haven't installed classy[{extra_required}], try doing `pip install classy[{extra_required}]`"
                raise ModuleNotFoundError(error_message)
            return decorated_arg(*args, **kwargs)

        return inner

    return closure
