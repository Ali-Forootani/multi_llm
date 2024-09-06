# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:37:58 2024

@author: forootan
"""

import torch
from transformers import (AutoModelForCausalLM,
                          TrainingArguments,
                          Trainer)
from transformers import LlamaTokenizer, LlamaForCausalLM
from pyprojroot import here
from prepare_training_data import prepare_cubetrianlge_qa_dataset