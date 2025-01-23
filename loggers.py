import os
import pytz
import wandb
import pandas as pd
from datetime import datetime

from os import listdir
from os.path import isfile, join

import common 

class WandBLogger:
    """WandB logger."""

    # def __init__(self, args, system_prompt, run_name):
    def __init__(self, args, system_prompt, planner_system_prompt, run_name):
        self.run_name = run_name
        self.logger = wandb.init(
            # project = "jailbreak-llms",
            project = args.project_name if args.project_name is not None else "RedAgent_debug",
            name = self.run_name,
            config = {
                "attack_model" : args.attack_model,
                "target_model" : args.target_model,
                "evaluator_model": args.evaluator_model,
                "keep_last_n": args.keep_last_n,
                "system_prompt": system_prompt,
                "planner_system_prompt": planner_system_prompt,  
                "index": args.index,
                "category": args.category,
                "goal": args.goal,
                "depth": args.depth,
                "width": args.width,
                "branching_factor": args.branching_factor,
                "target_str": args.target_str,
                "n_streams": args.n_streams,
                "add_score": args.add_score,
                "add_explanation": args.add_explanation,
                "judge_strategy": args.judge_strategy,
                "add_strategy": args.add_strategy,
                "self_reflection": args.self_reflection,
                "add_reflection": args.add_reflection,
                "reflection_frequency": args.reflection_frequency,
                "my_config": f"{args.add_score}_{args.add_explanation}_{args.add_strategy}_{args.self_reflection}"
            }
        )
        self.is_jailbroken = False
        self.query_to_jailbreak_trial = 0
        self.query_to_jailbreak = 0
        self.table = pd.DataFrame()
        self.batch_size = args.n_streams
        self.index = args.index
        self.goal = args.goal
        self.jailbreak_prompt = None
        self.jailbreak_response = None
        self.reflection_frequency = args.reflection_frequency
        self.jailbreak_success_score = args.jailbreak_success_score

    def log(self, trial_index: int, initial_plan, iteration: int, attack_list: list, response_list: list, judge_scores: list, explanations: list, on_topic_scores: list = None, conv_ids: list = None, parent_conv_ids: list = None, attack_strategy_list: list = None, reflection_list: list = None):
        # iteration is in range(1, attack_params['depth'] + 1)
        df = pd.DataFrame(attack_list)
        df["target_response"] = response_list
        df["judge_scores"] = judge_scores
        df["explanations"] = explanations
        df["on_topic_scores"] = on_topic_scores if on_topic_scores is not None else [-1 for a in judge_scores]
        df["trial_index"] = trial_index
        df["initial_plan"] = initial_plan
        df["iter"] = iteration
        
        df["conv_id"] = conv_ids if conv_ids is not None else [i+1 for i in range(len(response_list))]
        df["parent_conv_id"] = parent_conv_ids if parent_conv_ids is not None else [i+1 for i in range(len(response_list))]
        df["attack_strategy"] = attack_strategy_list if attack_strategy_list is not None else ["None" for i in range(len(response_list))]
        df["reflection"] = reflection_list if reflection_list is not None and len(reflection_list) > 0 else ["None" for i in range(len(response_list))]

        self.table = pd.concat([self.table, df])
        
        if iteration == 1:
            self.query_to_jailbreak_trial = 0
        self.query_to_jailbreak_trial += len(response_list)
        self.query_to_jailbreak += len(response_list)

        if any([score >= self.jailbreak_success_score for score in judge_scores]):
            # jailbreak_ind = judge_scores.index(10)
            jailbreak_ind  = next((i for i, score in enumerate(judge_scores) if score >= self.jailbreak_success_score), None)
            if not self.is_jailbroken:
                self.logger.log({"queries_to_jailbreak": self.query_to_jailbreak})
                self.is_jailbroken = True

            self.jailbreak_prompt = attack_list[jailbreak_ind]["prompt"]
            self.jailbreak_response = response_list[jailbreak_ind]

        self.logger.log({
            "trial_index":trial_index,
            "iteration":iteration,
            "judge_scores":judge_scores,
            "mean_judge_score_iter": sum(judge_scores) / len(judge_scores),
            "is_jailbroken":self.is_jailbroken,
            "max_judge_score":self.table["judge_scores"].max(),
            "jailbreak_prompt":self.jailbreak_prompt,
            "jailbreak_response":self.jailbreak_response,
            "data": wandb.Table(data = self.table)})

        self.print_iter_summary_stats(trial_index, iteration)

    def finish_trial(self, trial_index: int, plan, trajectory):
        self.print_trial_summary_stats(trial_index, plan, trajectory)

    def finish(self, trial_index: int):
        self.print_final_summary_stats(trial_index)
        self.logger.finish()

    def print_iter_summary_stats(self, trial_index, iter):
        df = self.table
        df = df[df['trial_index'] == trial_index]  
        mean_score_for_iter = df[df['iter'] == iter]['judge_scores'].mean()
        max_score_for_iter = df[df['iter'] == iter]['judge_scores'].max()
        
        num_total_jailbreaks = df[df['judge_scores'] >= self.jailbreak_success_score]['conv_id'].nunique()
        
        jailbreaks_at_iter = df[(df['iter'] == iter) & (df['judge_scores'] >= self.jailbreak_success_score)]['conv_id'].unique()
        prev_jailbreaks = df[(df['iter'] < iter) & (df['judge_scores'] >= self.jailbreak_success_score)]['conv_id'].unique()

        num_new_jailbreaks = len([cn for cn in jailbreaks_at_iter if cn not in prev_jailbreaks])

        print(f"{'='*14} Trial {trial_index} Iter {iter} SUMMARY STATISTICS {'='*14}")
        print(f"Mean/Max Score for iteration: {mean_score_for_iter:.1f}, {max_score_for_iter}")
        print(f"Number of New Jailbreaks: {num_new_jailbreaks}/{self.batch_size}")
        print(f"Total Number of Conv. Jailbroken: {num_total_jailbreaks}/{self.batch_size} ({num_total_jailbreaks/self.batch_size*100:2.1f}%)\n")

    def print_trial_summary_stats(self, trial_index, plan, trajectory):
        print(f"{'='*8} Trial {trial_index} SUMMARY STATISTICS {'='*8}")
        print(f"Index: {self.index}")
        print(f"Goal: {self.goal}")
        print(f"Plan from planner:\n{plan}\n")
        df = self.table
        df = df[df['trial_index'] == trial_index]  
        if self.is_jailbroken:
            num_total_jailbreaks = df[df['judge_scores'] >= self.jailbreak_success_score]['conv_id'].nunique()
            print(f"First Jailbreak: {self.query_to_jailbreak_trial} queries in trial {trial_index}") # jailbreak in a trial
            print(f"Total Number of Conv. Jailbroken: {num_total_jailbreaks}/{self.batch_size} ({num_total_jailbreaks/self.batch_size*100:2.1f}%)")
            print(f"Example Jailbreak PROMPT:\n\n{self.jailbreak_prompt}\n\n")
            print(f"Example Jailbreak RESPONSE:\n\n{self.jailbreak_response}\n\n\n")            
            
        else:
            print(f"No jailbreaks achieved in Trial {trial_index}.")
            max_score = df['judge_scores'].max()
            print(f"Max Score in trial {trial_index}: {max_score}")

        print(f"Trajectory:\n", trajectory)
        # self.table.to_parquet(common.STORE_FOLDER + '/' + f'iter_{common.ITER_INDEX}_df')

    def print_final_summary_stats(self, trial_index):
        print(f"{'='*8} FINAL SUMMARY STATISTICS {'='*8}")
        print(f"Index: {self.index}")
        print(f"Goal: {self.goal}")
        df = self.table
        if self.is_jailbroken:
            print(f"First Jailbreak: {self.query_to_jailbreak_trial} queries in trial {trial_index}")
            print(f"Total queries: {self.query_to_jailbreak}")          
        else:
            print("No jailbreaks achieved.")
            max_score = df['judge_scores'].max()
            print(f"Max Score in all trials: {max_score}")

        # self.table.to_parquet(common.STORE_FOLDER + '/' + f'iter_{common.ITER_INDEX}_df')
