# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535
unset ROCR_VISIBLE_DEVICES
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
OFFLOAD=${OFFLOAD:-False}
GPUS=${GPUS:-8}
N_ROLLOUT=${N_ROLLOUT:-8}
B=${B:-3}
RAY_DEBUG=${RAY_DEBUG:-0}
DEBUG=${DEBUG:-""}
DSET=${DSET:-"interaction"}
FORCE_THINKING=${FORCE_THINKING:-""}
EPOCHS=${EPOCHS:-15}
TP=${TP:-2}
QWEN=${QWEN:-2.5}
MAX_LEN_CTX=${MAX_LEN_CTX:-$((1024 * 4))} # max_model_len is determined by this which is max context len.
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-$MAX_LEN_CTX}
MAX_RESP_LEN=${MAX_RESP_LEN:-$MAX_LEN_CTX}
MAX_TOK=${MAX_TOK:-16384}
INSTRUCT=${INSTRUCT:-True}
MULTI_CONTEXT=${MULTI_CONTEXT:-False}
if [ $MULTI_CONTEXT == False ]; then
    BALANCE_BATCH=True
else
    BALANCE_BATCH=False
fi
DYNAMIC_BSZ=${DYNAMIC_BSZ:-$MULTI_CONTEXT}
if [ $QWEN -eq 3 ]; then
    if [ $INSTRUCT == True ]; then
        MODEL_NAME=Qwen/Qwen${QWEN}-${B}B
    else
        MODEL_NAME=Qwen/Qwen${QWEN}-${B}B-Base
    fi
else
    if [ $INSTRUCT == True ]; then
        MODEL_NAME=Qwen/Qwen${QWEN}-${B}B-Instruct
    else
        MODEL_NAME=Qwen/Qwen${QWEN}-${B}B
    fi
fi

# DEBUG=miss GPUS=2 MICRO_BATCH_SIZE=4 CUDA_VISIBLE_DEVICES="2,3" B=3 MINI_BATCH_SIZE=512 N_ROLLOUT=8 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DSET=interaction_belief B=7 N_ROLLOUT=2 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DSET=interaction_think B=7 N_ROLLOUT=2 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=belief_no_reinstruct2 DSET=interaction_belief B=7 N_ROLLOUT=2 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=thought_force FORCE_THINKING="Let's think step by step" DSET=interaction B=7 N_ROLLOUT=2 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=thought_force2 FORCE_THINKING="Let's think step by step before giving the query." DSET=interaction B=7 N_ROLLOUT=2 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=belief_simple FORCE_THINKING="Let\'s update the belief state first, and then use that belief to determine the best query." DSET=interaction_simple_belief B=7 N_ROLLOUT=2 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=q3_4b_belief_simple FORCE_THINKING="" MAX_LEN_CTX=16384 TP=1 DSET=interaction_simple_belief B=4 N_ROLLOUT=2 MICRO_BATCH_SIZE=2 QWEN=3 EPOCHS=150 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=q2_5_14b_belief_simple FORCE_THINKING="Let\'s update the belief state first, and then use that belief to determine the best query." B=14 N_ROLLOUT=2 MICRO_BATCH_SIZE=4 TP=4 EPOCHS=5000 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=debuggy INSTRUCT=False RAY_DEBUG=1 CUDA_VISIBLE_DEVICES="2" B=3 GPUS=1 MICRO_BATCH_SIZE=4 TRAIN_BATCH_SIZE=16 N_ROLLOUT=4 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=test_base DSET=interaction_base_base INSTRUCT=False B=7 GPUS=4 MICRO_BATCH_SIZE=4 N_ROLLOUT=1 TRAIN_BATCH_SIZE=16 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=q2_5_7b_base DSET=interaction_base_base INSTRUCT=False B=7 GPUS=4 MICRO_BATCH_SIZE=4 N_ROLLOUT=2 EPOCHS=5000 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=debug DSET=interaction_base INSTRUCT=True B=3 GPUS=2 CUDA_VISIBLE_DEVICES="3,5" MICRO_BATCH_SIZE=2 N_ROLLOUT=1 TRAIN_BATCH_SIZE=4 TP=1 MULTI_CONTEXT=True bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# added dset for debug base_one_val
# with dynamic bsz micro bsz doesn't do anything.
# DEBUG=q2_5_14b_mt_belief_base DSET=interaction_base INSTRUCT=False B=14 GPUS=8 MICRO_BATCH_SIZE=4 N_ROLLOUT=2 TP=4 MULTI_CONTEXT=True TRAIN_BATCH_SIZE=16 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=q2_5_14b_mc_belief_base DSET=interaction_base_base INSTRUCT=False B=14 GPUS=4 N_ROLLOUT=2 TP=1 MULTI_CONTEXT=True TRAIN_BATCH_SIZE=64 MAX_TOK=4096 MAX_LEN_CTX=2048 EPOCHS=5000 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# DEBUG=q2_5_14b_sc_belief_base DSET=interaction_base_base INSTRUCT=False B=14 GPUS=4 N_ROLLOUT=2 TP=1 MULTI_CONTEXT=False DYNAMIC_BSZ=True TRAIN_BATCH_SIZE=64 MAX_TOK=4096 MAX_RESP_LEN=3584 MAX_PROMPT_LEN=512 MAX_LEN_CTX=4096 EPOCHS=5000 bash examples/sglang_multiturn/run_qwen2.5-xb_combolock_multiturn_w_interaction.sh
# added use_dynamic_bsz if multi context
# added balance_batch False if multi context to avoid 
# I changed 0.7 mem to 0.5 just when switching to 7 B instead of 3 B model.

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='combolock_multiturn_grpo_w_interaction' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESP_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_LEN_CTX \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOK \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$N_ROLLOUT \
    "actor_rollout_ref.rollout.multi_turn.force_thinking='$FORCE_THINKING'" \
    actor_rollout_ref.rollout.is_instruct_model=$INSTRUCT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.use_dynamic_bsz=$DYNAMIC_BSZ \
    trainer.balance_batch=$BALANCE_BATCH \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl-tests' \
    trainer.experiment_name=qwen${QWEN}-${B}b_function_rm-combolock-sgl-multi-w-$DSET-n$N_ROLLOUT-2-$DEBUG \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    data.train_files=$PROJECT_DIR/../data/multi_turn_combo_lock_$DSET/train.parquet \
    data.val_files=$PROJECT_DIR/../data/multi_turn_combo_lock_$DSET/test.parquet \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/combolock_interaction_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.multi_context=$MULTI_CONTEXT \
    reward_model.reward_manager=multiturn \
    trainer.log_val_generations=10 \
    trainer.test_freq=5 \
    +custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/multiturn.py \
    trainer.total_epochs=$EPOCHS $@