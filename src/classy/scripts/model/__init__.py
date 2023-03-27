import os

import transformers

# setting the transformers logging level to error with their own function
transformers.logging.set_verbosity_error()

# todo: we should analyze this error a little bit. Gigi can be useful here.
#  According to : https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
#  We should check if we really need FastTokenizers instead of the plain tokenizer.
# turning off the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"
