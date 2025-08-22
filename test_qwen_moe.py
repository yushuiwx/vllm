from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="/mnt/msranlp/xun/reasoning/MoE-TTS/results/training/countdown/sft-hint-4-cd-activte85M-all385M-e8t1-qwen-moe-topk-4xH100/checkpoint-1000")

# 获取模型引用
model = llm.model_executor.driver_worker.model_runner.model.model

# 清空之前的 router logits
model.clear_router_logits()

# 生成文本
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(max_tokens=10)
outputs = llm.generate(prompts, sampling_params)

# 获取 router logits
all_router_logits = model.get_router_logits()

# all_router_logits 的结构：
# - 外层 list: 每次 forward pass（可能多次，取决于生成长度）
# - 中层 list: 每一层
# - 内层 tensor: [batch_size * seq_len, num_experts]

print(f"Total forward passes: {len(all_router_logits)}")
if all_router_logits:
    print(f"Number of layers: {len(all_router_logits[0])}")
    print(f"First layer router logits shape: {all_router_logits[0][0].shape}")

# 处理成每个 token 的格式（如果需要）
def reorganize_router_logits(all_router_logits):
    """
    将 router logits 重组为每个 token 的格式
    """
    token_router_logits = []
    for forward_pass_logits in all_router_logits:
        # forward_pass_logits: [num_layers, batch_size * seq_len, num_experts]
        num_layers = len(forward_pass_logits)
        if num_layers > 0:
            seq_len = forward_pass_logits[0].shape[0]
            for token_idx in range(seq_len):
                token_logits = []
                for layer_idx in range(num_layers):
                    token_logits.append(forward_pass_logits[layer_idx][token_idx])
                token_router_logits.append(token_logits)
    return token_router_logits

# 获取每个 token 的 router logits
token_wise_logits = reorganize_router_logits(all_router_logits)
print(f"Total tokens generated: {len(token_wise_logits)}")