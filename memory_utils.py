import re
import json
from collections import defaultdict
from functools import singledispatch

def convert_str_to_dict(s):
    pattern = r'category: "(?P<category>.+?)", question: "(?P<question>.+?)", prompt: "(?P<prompt>.+?)", strategy: "(?P<strategy>.+?)", score: (?P<score>\d+)'
    data = [match.groupdict() for match in re.finditer(pattern, s, re.DOTALL)]
    for entry in data:
        entry['score'] = int(entry['score'])
    return data

def convert_dict_to_str(data):
    return '\n'.join(f'category: "{entry["category"]}", question: "{entry["question"]}", prompt: "{entry["prompt"]}", strategy: "{entry["strategy"]}", score: {entry["score"]}' for entry in data)

def handle_file_path(func):
    def wrapper(memory_path, *args, **kwargs):
        memory_list = load_memory(memory_path)
        result = func(memory_list, *args, **kwargs)
        save_memory(memory_path, result)
        return result
    return wrapper

def load_memory(memory_path):
    try:
        with open(memory_path, 'r', encoding='utf-8') as f:
            if memory_path.endswith('.txt'):
                return convert_str_to_dict(f.read())
            elif memory_path.endswith('.json'):
                return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {memory_path}")
        return []

def save_memory(memory_path, memory_list):
    try:
        with open(memory_path, 'w', encoding='utf-8') as f:
            if memory_path.endswith('.txt'):
                f.write(convert_dict_to_str(memory_list))
            elif memory_path.endswith('.json'):
                json.dump(memory_list, f, indent=4)
    except IOError as e:
        print(f"Error saving file {memory_path}: {e}")


@singledispatch
def sort_memories(memory):
    raise NotImplementedError("Unsupported type")

@sort_memories.register(list)
def _sort_memories_list(memory_list):
    print(f"Sorting memories, memory list length: {len(memory_list)}")
    memory_list.sort(key=lambda x: (x.get('category', ''), x.get('question', ''), x['score']), reverse=True)
    return memory_list

@sort_memories.register(str)
@handle_file_path
def _sort_memories_from_path(memory_list):
    return _sort_memories_list(memory_list)


@singledispatch
def filter_memory(memory, remain_num):
    raise NotImplementedError("Unsupported type")

@filter_memory.register(list)
def _filter_memory_list(memory_list, remain_num):   
    if remain_num <= 0:
        return []
    if len(memory_list) <= remain_num:
        return memory_list
    print(f"Filtering memories, remain {remain_num} of {len(memory_list)}")

    selected_memories = memory_list
    if memory_list[0].get('category'):
        category_groups = defaultdict(list)
        for memory in memory_list:
            category_groups[memory['category']].append(memory)
        if len(category_groups) > remain_num:
            seen_prompts = set()
            unique_memories = []
            for memory in memory_list:
                if memory['prompt'] not in seen_prompts:
                    seen_prompts.add(memory['prompt'])
                    unique_memories.append(memory)
            memory_list = unique_memories

        
        category_groups = defaultdict(list)
        for memory in memory_list:
            category_groups[memory['category']].append(memory)
        for category in category_groups:
            category_groups[category] = sorted(category_groups[category], key=lambda x: -x['score'])
        selected_memories = sum((memories[:max(remain_num // len(category_groups), 1)] for memories in category_groups.values()), [])

    
    if len(selected_memories) > remain_num:
        selected_memories.sort(key=lambda x: -x['score'])
        selected_memories = selected_memories[:remain_num]
    return selected_memories

@filter_memory.register(str)
@handle_file_path
def _filter_memory_from_path(memory_list, remain_num):
    return _filter_memory_list(memory_list, remain_num)


@singledispatch
def deduplicate_memory(memory):
    raise NotImplementedError("Unsupported type")

@deduplicate_memory.register(list)
def _deduplicate_memory_list(memory_list):
    if not memory_list[0].get('question'):
        return memory_list
    
    best_memories = {}
    for memory in memory_list:
        current_question = memory['question']
        if current_question not in best_memories or memory['score'] > best_memories[current_question]['score']:
            best_memories[current_question] = memory
    print("Deduplicating memories, before: {}, after: {}".format(len(memory_list), len(best_memories)))
    return list(best_memories.values())

@deduplicate_memory.register(str)
@handle_file_path
def _deduplicate_memory_from_path(memory_list):
    return _deduplicate_memory_list(memory_list)



@singledispatch
def update_memory(memory, new_entries, memory_limit):
    raise NotImplementedError("Unsupported type")

@update_memory.register(list)
def _update_memory_list(memory_list, new_entries, memory_limit):
    print(f"Updating memories, memory list length: {len(memory_list)}, adding {len(new_entries)} new entries")
    if len(new_entries) > memory_limit:
        new_entries = new_entries[:memory_limit]
    if len(memory_list) + len(new_entries) > memory_limit:
        memory_list = filter_memory(memory_list, memory_limit - len(new_entries))
    new_entries = '\n'.join(new_entries)
    new_entries = convert_str_to_dict(new_entries)
    memory_list.extend(new_entries)
    return memory_list

@update_memory.register(str)
@handle_file_path
def _update_memory_path(memory_list, new_entries, memory_limit):
    return _update_memory_list(memory_list, new_entries, memory_limit)