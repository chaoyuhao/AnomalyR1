# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
#     "The answer should contains only letters A, B, C, D, or a bounding box in the format [x1, y1, x2, y2]."
# )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a choice question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer."
    "Respond with your reasoning in <think> </think> tags "
    "followed by a single letter answer in <answer> </answer> tags."
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        # FIXME
        # This is only for Grounding task
        # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        QUESTION_TEMPLATE = "{Question} " + SYSTEM_PROMPT

        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            
            # 检查图像尺寸并在必要时压缩
            max_width, max_height = 1000, 800  # 设置最大尺寸
            if image.width > max_width or image.height > max_height:
                # 计算宽高比
                old_width, old_height = image.width, image.height
                ratio = min(max_width / image.width, max_height / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                # 调整图像大小
                image = image.resize(new_size, Image.LANCZOS)
                if os.getenv("DEBUG_MODE") == "true":
                    print(f"Resized image {image_path} from {old_width}x{old_height} to {new_size[0]}x{new_size[1]}")
        else:
            image = None
        

        return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''
def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def extract_choice(text):
    """从文本中智能提取选择题答案（A, B, C, D）"""
    if not text:
        return None
        
    # 1. 清理和标准化文本
    text = re.sub(r'\s+', ' ', text)  # 规范化空格
    
    if len(text) < 3:
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号        
        return text[0] if text else None

    # 2. 选项不应该前后有字母
    choices = re.findall(r'(?<![A-Z])([A-D])(?![A-Z])', text)
    choices = re.findall(r'(?<![a-z])([A-D])(?![a-z])', text)

    if not choices:
        return None

    # 3. 如果只有一个选项，直接返回
    if len(choices) == 1:
        return choices[0]

    # 4. 如果有多个选项，使用启发式规则
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 关键词周围的选项加分
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # 获取每个选项的上下文（前后20个字符）
    for choice in choices:
        pos = text.rfind(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # 关键词加分
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # 如果选项靠近文本末尾则加分（通常是最终答案）
        if pos > len(text) * 0.7:  # 在文本后30%
            choice_scores[choice] += 2

        # 如果后面跟着标点符号则加分
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # 返回得分最高的选项
    return max(choice_scores.items(), key=lambda x: x[1])[0]

def roam_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has the correct answer (A, B, C, or D)
    and if the thinking process supports the correct answer."""
    # 仅提取模型回答部分，忽略用户输入
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        
        try:
            # First try to find if there's a proper format with <think> and <answer> tags
            match = re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', content, re.DOTALL)
            if match:
                
                # 查找所有<answer>标签对
                content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)
                content_think_matches = re.findall(think_tag_pattern, content, re.DOTALL)
                
                
                # 获取答案部分
                if content_answer_matches:
                    content_answer = content_answer_matches[-1].strip()
                    answer_choice = extract_choice(content_answer)
                else:
                    answer_choice = None
                
                # 分析思维链部分
                if content_think_matches:
                    thinking = content_think_matches[-1].strip()
                    thinking_choice = extract_choice(thinking)
                else:
                    thinking_choice = None

                if thinking_choice == None and answer_choice == None:
                    reward = 0.0
                elif thinking_choice == None and answer_choice == sol:
                    reward = 0.8
                elif thinking_choice == None and answer_choice != sol:
                    reward = 0.05
                elif thinking_choice == sol and answer_choice == sol:
                    reward = 1.0
                elif thinking_choice != sol and answer_choice != sol and thinking_choice == answer_choice:
                    reward = 0.1
                elif thinking_choice != sol and answer_choice != sol and thinking_choice != answer_choice:
                    reward = 0.0
                else:
                    reward = 0.0
            else:
                reward = 0.0
                thinking_choice = None
                answer_choice = None
                content_answer_matches = None
                content_think_matches = None
        

        except Exception as e:
            content_answer_matches = f"Answer extract error: {str(e)}"
            
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Choice reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Extracted answer: {answer_choice}\n")
                f.write(f"Thinking conclusion: {thinking_choice}\n")
                f.write(f"GPT Answer: {content_answer_matches}\n")
                f.write(f"GPT Thinking: {content_think_matches}\n")
                f.write(f"Correct Answer: {sol}\n")
        
        rewards.append(reward)
    return rewards

def choice_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has the correct answer (A, B, C, or D)
    and if the thinking process supports the correct answer."""
    # 仅提取模型回答部分，忽略用户输入
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        
        try:
            # First try to find if there's a proper format with <think> and <answer> tags
            match = re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', content, re.DOTALL)
            if match:
                
                # 查找所有<answer>标签对
                content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)
                content_think_matches = re.findall(think_tag_pattern, content, re.DOTALL)
                
                
                # 获取答案部分
                if content_answer_matches:
                    content_answer = content_answer_matches[-1].strip()
                    answer_choice = extract_choice(content_answer)
                else:
                    content_answer = None
                    answer_choice = None
                
                if answer_choice == sol:
                    reward = 1.0
            
            else:
                reward = 0.0
                thinking_choice = None
                answer_choice = None
                content_answer_matches = None
                content_think_matches = None
        

        except Exception as e:
            content_answer_matches = f"Answer extract error: {str(e)}"
            
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Choice easy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Extracted answer: {answer_choice}\n")
                f.write(f"GPT Answer: {content_answer_matches}\n")
                f.write(f"GPT Thinking: {content_think_matches}\n")
                f.write(f"Correct Answer: {sol}\n")
        
        rewards.append(reward)
    return rewards

def format_easy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has the correct answer (A, B, C, or D)
    and if the thinking process supports the correct answer."""
    # 仅提取模型回答部分，忽略用户输入
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        
        try:
            # First try to find if there's a proper format with <think> and <answer> tags
            match = re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', content, re.DOTALL)
            if match:
                reward = 1.0
            else:
                reward = 0.0
        

        except Exception as e:
            content_answer_matches = f"Answer extract error: {str(e)}"
            
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} fmt reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Correct Answer: {sol}\n")
        
        rewards.append(reward)
    return rewards

def choice_format_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has the correct answer (A, B, C, or D) and correct format
    and if the thinking process supports the correct answer."""
    # 仅提取模型回答部分，忽略用户输入
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'
    pattern = r'<think>.*?</think>.*?<answer>.*?</answer>'
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        if re.search(pattern, content, re.DOTALL):
            reward += 0.1

        try:
            # 查找所有<answer>标签对
            content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)
            content_think_matches = re.findall(think_tag_pattern, content, re.DOTALL)
            
            # 获取答案部分
            if content_answer_matches:
                content_answer = content_answer_matches[-1].strip()
                answer_choice = extract_choice(content_answer)
            else:
                answer_choice = None
            
            # 分析思维链部分
            if content_think_matches:
                thinking = content_think_matches[-1].strip()
                thinking_choice = extract_choice(thinking)
            else:
                thinking_choice = None

            if thinking_choice == None and answer_choice == None:
                reward = 0.0
            elif thinking_choice == None and answer_choice == sol:
                reward = 0.7
            elif thinking_choice == None and answer_choice != sol:
                reward = 0.1
            elif thinking_choice == sol and answer_choice == sol:
                reward = 1.0
            elif thinking_choice != sol and answer_choice != sol and thinking_choice == answer_choice:
                reward = 0.2
            elif thinking_choice != sol and answer_choice != sol and thinking_choice != answer_choice:
                reward = 0.0
            else:
                reward = 0.0

        except Exception as e:
            content_answer_matches = f"Answer extract error: {str(e)}"
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format+0.1 Choice reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Extracted answer: {answer_choice}\n")
                f.write(f"Thinking conclusion: {thinking_choice}\n")
                f.write(f"GPT Answer: {content_answer_matches}\n")
                f.write(f"GPT Thinking: {content_think_matches}\n")
                f.write(f"Correct Answer: {sol}\n")
        
        rewards.append(reward)
    return rewards

def format_zero_reward(completions, **kwargs):
    """Reward function that checks if the completion has the correct format with <think> and <answer> tags."""
    # Extract content properly from different completion structures
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content in contents:
        reward = 0.0
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format zero reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the correct format with <think> and <answer> tags."""
    # Extract content properly from different completion structures
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # Pattern to check for both <think> and <answer> tags in proper order
    pattern = r'<think>.*?</think>.*?<answer>.*?</answer>'
    
    for content in contents:
        # Check if the basic format pattern exists in the content
        reward = 0
        # if re.search(pattern, content, re.DOTALL):
            

        #     reward = 0.1

        # else:
        #     reward = 0.0
        
        

        rewards.append(reward)
        
        # Add debugging output if needed
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                has_think = "<think>" in content and "</think>" in content
                has_answer = "<answer>" in content and "</answer>" in content
                f.write(f"Has think tags: {has_think}, Has answer tags: {has_answer}\n")
    
    return rewards

def process_completion(completion):
    # 只保留思考和答案部分
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    
    if think_match and answer_match:
        return f"<think>{think_match.group(1)}</think><answer>{answer_match.group(1)}</answer>"
    return completion

reward_funcs_registry = {
    "accuracy": choice_reward,
    "format": roam_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
