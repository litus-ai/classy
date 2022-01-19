HELP_MODEL_PATH = """
    The model you want to use for the demo. Can be
    1) the experiment name: "my_experiment" and classy will automatically
    look for the most recent run and the best checkpoint for that run under
     "experiments/my_experiment". 2) experiment directory path:
     "experiments/my_experiment" and classy will automatically look for the most
    recent run and the best checkpoint of that run under the provided model directory.
    3) experiment directory comprising of date and hour (i.e. specific run):
    "experiments/my_experiments/20-10-2021/15-23-58" and classy will look for the best
    checkpoint for that specific run 4) experiment specific checkpoint:
    "experiments/my_experiments/20-10-2021/15-23-58/checkpoints/last.ckpt.
"""


HELP_TOKEN_BATCH_SIZE = "The maximum amount of tokens in a batch."

HELP_FILE_PATH = """
    Optional. If specified the evaluation will be performed on this file. Otherwise, classy will try to infer
    the file_path from the training configuration. Either by searching under dataset_path/test.data_format where
    "dataset_path" is the directory passed at training time; or under the "model_path" directory if you passed only
    one file at training time.
"""

HELP_EVALUATE = """
    Path to evaluation config to use.
"""

HELP_PREDICTION_PARAMS = """
    Path to prediction params.
"""

HELP_TASKS = """
    One of the tasks that classy supports [sequence, sentence-pair, token, qa, generation].
"""
