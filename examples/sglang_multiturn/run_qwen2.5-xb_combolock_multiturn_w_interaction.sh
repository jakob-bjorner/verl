# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535
unset ROCR_VISIBLE_DEVICES
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
MINI_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
OFFLOAD=${OFFLOAD:-False}
GPUS=${GPUS:-8}
N_ROLLOUT=${N_ROLLOUT:-8}
B=${B:-3}
RAY_DEBUG=${RAY_DEBUG:-0}
DEBUG=${DEBUG:-""}
# DEBUG=miss GPUS=2 MICRO_BATCH_SIZE=4 CUDA_VISIBLE_DEVICES="2,3" B=3 MINI_BATCH_SIZE=512 N_ROLLOUT=8 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# I changed 0.7 mem to 0.5 just when switching to 7 B instead of 3 B model.
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='combolock_multiturn_grpo_w_interaction' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=$((1024 * 2)) \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-${B}B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$N_ROLLOUT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl-tests' \
    trainer.experiment_name=qwen2.5-${B}b_function_rm-combolock-sgl-multi-w-interaction-n$N_ROLLOUT-1-$DEBUG \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.log_val_generations=10 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    data.train_files=/nas/ucb/jbjorner3/data/multi_turn_combo_lock_interaction/train.parquet \
    data.val_files=/nas/ucb/jbjorner3/data/multi_turn_combo_lock_interaction/test.parquet \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/combolock_interaction_config.yaml" \
    reward_model.reward_manager=multiturn \
    +custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/multiturn.py \
    trainer.total_epochs=15 $@