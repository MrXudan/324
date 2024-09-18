#coding=utf-8
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 图像文件路径列表
IMAGES = [
            "/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg",  # 本地图片路径
            ]

# 模型名称或路径
MODEL_NAME = "/root/ld/ld_model_pretrained/Minicpmv2_6"  # 本地模型路径或Hugging Face模型名称

# 打开并转换图像
image = Image.open(IMAGES[0]).convert("RGB")

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 初始化语言模型
llm = LLM(model=MODEL_NAME,
                   gpu_memory_utilization=1,  # 使用全部GPU内存
                              trust_remote_code=True,
                                         max_model_len=2048)  # 根据内存状况可调整此值

# 构建对话消息
messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + '请描述这张图片'}]

# 应用对话模板到消息
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 设置停止符ID
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
#stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
# 2.6 
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

# 设置生成参数
sampling_params = SamplingParams(
            stop_token_ids=stop_token_ids,
                # temperature=0.7,
                    # top_p=0.8,
                        # top_k=100,
                            # seed=3472,
                                max_tokens=1024,
                                    # min_tokens=150,
                                        temperature=0,
                                            use_beam_search=True,
                                                # length_penalty=1.2,
                                                    best_of=3)

# 获取模型输出
outputs = llm.generate({
        "prompt": prompt,
            "multi_modal_data": {
                        "image": image
                            }
            }, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
                                  