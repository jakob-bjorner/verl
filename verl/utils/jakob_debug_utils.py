__all__ = []
# # from .protocol import DataProto
# from .tokenizer import hf_tokenizer

# def print_batch(data_og: DataProto):
"""
    first job is to ensure nothing has shifted, make sure that the advantages is properly masked and the input ids aren't revealing anything
    data_og.batch["advantages"][batch.batch["loss_mask"][2048:]]
    batch = data_og
    (data_og.batch["input_ids"][data_og.batch["loss_mask"] == 1] == data_og.batch["input_ids"][0,0]).any() # this should be false
    print([(tokenizer.decode(data_og.batch["input_ids"][i][data_og.batch["loss_mask"][i] == 0], skip_special_tokens=True) + "\n==================\n" + tokenizer.decode(data_og.batch["input_ids"][i][data_og.batch["loss_mask"][i] == 1], skip_special_tokens=True)) for i in range(data_og.batch.batch_size[0])][-2])
     # this is only the text from the prompt
    data_og.non_tensor_batch.keys()
    ['data_source', 'ability', 'reward_model', 'extra_info', 'index', 'uid', 'messages', 'reward_scores', 'request_ids', 'context_indices', 'to_log_stats']
    data_og.non_tensor_batch['to_log_stats'][0].keys()
    ['trajectory_info', 'prompt_tokens_per_belief_message', 'prompt_tokens_per_action_message', 'tokens_per_action_message', 'tokens_per_belief_generation_message', 'tokens_per_belief_state_message', 'run_success', 'run_attempts', 'run_completion', 'belief_gen_failures']
    data_og.batch["input_ids"][0][data_og.batch["attention_mask"][0]]
    print("\n==========================================\n".join(tokenizer.batch_decode(data_og.batch['input_ids'][[i for (i, idx) in enumerate(data_og.non_tensor_batch["request_ids"]) if idx == "2f07cf98-d538-4196-98b4-3a4cebe78fac"]], skip_special_tokens=True)))
     # single sample print
    {j: i for i, j in enumerate(data_og.non_tensor_batch['request_ids'])}
    [(data_og.non_tensor_batch['to_log_stats'][i]['trajectory_info']["feedback_hist"], data_og.batch['advantages'][i,0].item(), data_og.non_tensor_batch["reward_scores"][i]['interaction_reward'][0], data_og.non_tensor_batch['to_log_stats'][i]['trajectory_info']["invalid_format_errors"], data_og.non_tensor_batch['to_log_stats'][i]['prompt_tokens_per_action_message'].__len__(), j, data_og.non_tensor_batch['uid'][i], i) for j, i in {j: i for i, j in enumerate(data_og.non_tensor_batch['request_ids'])}.items()]
    # are there more generation failures in the successful runs? (potentially because the successful runs go longer??)
    # like is there more positive or negative gradient when the model outputs some an imparsable action or has repeats.
    sum([(data_og.batch['advantages'][i,0].item()* data_og.non_tensor_batch['to_log_stats'][i]['trajectory_info']["invalid_format_errors"]) for i in {j: i for i, j in enumerate(data_og.non_tensor_batch['request_ids'])}.values()])
    # check weighted sum of advantage across traces with invalid_format_errors see if it is positive or negative indicating a particular preference for this activity... it was negative indicating this behavior is primarily in decline hopefully...
    should check sequences which don't terminate, and ensure they recieve low reward. will do after launch single batch 7 b instruct multi context run.
    data_og.batch["loss_mask"][:, -data_og.batch["response_mask"].size(1):]

    compute_grpo_outcome_advantage(token_level_rewards=data_og.batch['token_level_rewards'],
                                   response_mask=data_og.batch["loss_mask"][:, -data_og.batch["response_mask"].size(1):],
                                   index=data.non_tensor_batch["uid"],
                                   request_ids=data.non_tensor_batch["request_ids"],
                                   norm_adv_by_std_in_grpo=True,)
check full batches for advantage issues
save data_og and anything else (for later testing in quick_rollout.ipynb)
ensure on next step format_errors goes down (forgot to check this, but I checked it in a different piece of code, but now I don't know what to compare the number to when I go to try it out in a notebook. I should relaunch??)


output2 = self.actor_module(
                    input_ids=input_ids_rmpad[:,452:1015],
                    attention_mask=None,
                    position_ids=position_ids_rmpad[:,452:1015],
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
"""
    # ...
    # tokenizer = hf_tokenizer("qwen")
    # print(batch)
    # the batch will have some content, and we should print this content in an organized fashion. 
    # We could just assume that the data is organized with the non_tensor_batch... I should interactively debug with this thing, and then import this function ...
