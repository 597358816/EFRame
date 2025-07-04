set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=RFR-DAPO \
    trainer.n_gpus_per_node=8
