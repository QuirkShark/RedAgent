import os
import sys
import shutil
from memory_utils import load_memory, save_memory
import pandas as pd
import re


for index in range(50):
    os.system(f'python main.py --attack-model gpt-3.5-turbo --target-model llama-2 --evaluator-model gpt-4-1106-preview --planner-model gpt-3.5-turbo --store-folder ./result --branching-factor 1 --depth 4 --project-name RedAgent_Llama-2-7b-chat-hf --num-trial 5 --judge-strategy --add-score --add-strategy --jailbreak-success-score 8 --memory-limit 25 --index {index}')