#coding=utf-8
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'#减少内存碎片

with open('/root/324/box.prompt', 'r', encoding='utf-8') as f:  #单次prompt,用于明确工作内容和回复格式
    prompt_once = f.read()

# 图像文件路径列表
IMAGES = [
            "/root/324/prompt.png",  # 本地图片路径
            ]

# 模型名称或路径
MODEL_NAME = "/root/MiniCPM-V-2_6"  # 本地模型路径或Hugging Face模型名称
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 初始化语言模型
llm = LLM(model=MODEL_NAME,
                   gpu_memory_utilization=1,  # 使用全部GPU内存
                              trust_remote_code=True,
                                         max_model_len=2048)  # 根据内存状况可调整此值

# 打开并转换图像
image = Image.open(IMAGES[0]).convert("RGB")

def assist(talk):
    # 构建对话消息
    messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + talk}]
    #messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + '请描述这张图片'}]
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
assist(prompt_once)
while True:
    order = input("请输入指令，输入“退出”来终止>>")
    if order == '退出':
        print('Bye!')
        break
    assist(order)
