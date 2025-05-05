import argparse
import json
import os
import random
import re
import torch
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append("..")
from helper.summary import caculate_accuracy_mmad

# 导入必要的库
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import logging
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# 设置日志记录
log_dir = "inference_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"thinkad_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ThinkAD-Evaluation")

# class ThinkADQuery:
#     def __init__(self, image_path, text_gt, tokenizer, model, processor, 
#                  few_shot=[], domain_knowledge=None, args=None):
#         self.image_path = image_path
#         self.text_gt = text_gt
#         self.tokenizer = tokenizer
#         self.model = model
#         self.processor = processor
#         self.few_shot = few_shot
#         self.domain_knowledge = domain_knowledge
#         self.args = args
        
#     def parse_conversation(self, text_gt):
#         """从MMAD数据集格式中提取问题和答案"""
#         questions = []
#         answers = []
#         question_types = []
        
#         # 提取对话部分
#         keyword = "conversation"
#         for key in text_gt.keys():
#             if key.startswith(keyword):
#                 conversation = text_gt[key]
#                 for i, qa in enumerate(conversation):
#                     # 打乱选项顺序
#                     options_items = list(qa['Options'].items())
#                     random.shuffle(options_items)
                    
#                     # 重建选项文本
#                     options_text = ""
#                     new_answer_key = None
#                     for new_key, (original_key, value) in enumerate(options_items):
#                         options_text += f"{chr(65 + new_key)}. {value}\n"  # A, B, C...
#                         if qa['Answer'] == original_key:
#                             new_answer_key = chr(65 + new_key)
                    
#                     option_dict = {chr(65 + new_key): value for new_key, (original_key, value) in enumerate(options_items)}
                    
#                     # 构建问题
#                     questions_text = qa['Question']
#                     questions.append({
#                         "text": f"{questions_text} \n{options_text}",
#                         "options": option_dict,
#                     })
                    
#                     # 记录答案和问题类型
#                     answers.append(new_answer_key)
#                     question_types.append(qa.get('type', 'unknown'))
#                 break
                
#         return questions, answers, question_types
    
#     def generate_answer(self):
#         """为问题生成答案"""
#         questions, answers, question_types = self.parse_conversation(self.text_gt)
#         if not questions or not answers:
#             return questions, answers, None, question_types
            
#         gpt_answers = []
#         history_text = ""
        
#         # 如果有参考图像
#         reference_text = ""
#         if self.few_shot:
#             reference_text += f"Following is {len(self.few_shot)} normal sample image(s) for reference:\n<|image_pad|>\n"
        
#         # 如果有领域知识
#         domain_knowledge_text = ""
#         if self.domain_knowledge:
#             domain_knowledge_text = f"Domain knowledge about this object: {self.domain_knowledge}\n"
        
#         # 构建提示
#         base_prompt = domain_knowledge_text + reference_text
        
#         # 对每个问题生成答案
#         for i, question in enumerate(questions):
#             # 构建最终提示
#             full_prompt = f"{base_prompt}Test image: <|image_pad|>\n\n{question['text']}\n\nFirst output the thinking process in <think> </think> tags and then output the final answer letter in <answer> </answer> tags."
            
#             # 准备输入
#             images = []
#             if self.few_shot:
#                 for path in self.few_shot:
#                     ref_image = Image.open(path).convert("RGB")
#                     # Resize reference images too
#                     width, height = ref_image.size
#                     if width > 1000 or height > 800:
#                         ratio = min(1000 / width, 800 / height)
#                         new_width = int(width * ratio)
#                         new_height = int(height * ratio)
#                         ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
#                     images.append(ref_image)
                    
#             # 添加测试图像# 添加测试图像
#             test_image = Image.open(self.image_path).convert("RGB")
#             # Resize image to maximum 1000x800 while preserving aspect ratio
#             width, height = test_image.size
#             if width > 1000 or height > 800:
#                 ratio = min(1000 / width, 800 / height)
#                 new_width = int(width * ratio)
#                 new_height = int(height * ratio)
#                 test_image = test_image.resize((new_width, new_height), Image.LANCZOS)

#             images.append(test_image)
            
#             # 处理输入
#             inputs = self.processor(
#                 text=full_prompt,
#                 images=images,
#                 return_tensors="pt"
#             ).to(self.model.device)
            
#             # 生成回答
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=1024,
#                     do_sample=False,
#                     use_cache=True,
#                     temperature=None,
#                     top_p=None,
#                     top_k=None
#                 )
                
#             # 解码输出
#             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # 修改: 使用rfind查找最后出现的标签对，而不是使用正则表达式
#             # 这样可以避免匹配到提示中的标签
#             last_think_start = response.rfind("<think>")
#             last_think_end = response.rfind("</think>")
            
#             if last_think_start != -1 and last_think_end != -1 and last_think_start < last_think_end:
#                 thinking = response[last_think_start + 7:last_think_end].strip()
#             else:
#                 thinking = ""
            
#             last_answer_start = response.rfind("<answer>")
#             last_answer_end = response.rfind("</answer>")
            
#             if last_answer_start != -1 and last_answer_end != -1 and last_answer_start < last_answer_end:
#                 answer_text = response[last_answer_start + 8:last_answer_end].strip()
#             else:
#                 answer_text = ""
            
#             # 解析选项字母
#             answer_letter = extract_answer_letter(answer_text, question['options'])
            
#             if not answer_letter:
#                 # 如果未能提取到字母，设为X（表示无法确定）
#                 answer_letter = "X"
                
#             gpt_answers.append(answer_letter)
            
#             # 日志
#             logger.info(f"Question {i+1}: {question['text']}")
#             # logger.info(f"Thinking: {thinking[:100]}..." if len(thinking) > 100 else f"Thinking: {thinking}")
#             logger.info(f"Thinking: {thinking}")
#             logger.info(f"Answer: {answer_letter} (Raw: {answer_text})")
#             logger.info(f"Correct answer: {answers[i]}")
#             logger.info("-" * 50)
            
#         return questions, answers, gpt_answers, question_types


def extract_answer_letter(answer_text, options):
    """Extract the answer letter from model output"""
    if not answer_text:
        return "X"
        
    # Try to extract letter directly
    letter_match = re.search(r'([A-E])', answer_text)
    if letter_match:
        return letter_match.group(1)
    
    # Try to match by option text
    for key, value in options.items():
        if value.lower() in answer_text.lower():
            return key
            
    # Default if no match
    return "X"

def parse_conversation(text_gt):
    """From MMAD dataset format, extract questions and answers"""
    questions = []
    answers = []
    question_types = []
    
    # Extract conversation part
    keyword = "conversation"
    for key in text_gt.keys():
        if key.startswith(keyword):
            conversation = text_gt[key]
            for i, qa in enumerate(conversation):
                # Shuffle options
                options_items = list(qa['Options'].items())
                random.shuffle(options_items)
                
                # Rebuild options text
                options_text = ""
                new_answer_key = None
                for new_key, (original_key, value) in enumerate(options_items):
                    options_text += f"{chr(65 + new_key)}. {value}\n"  # A, B, C...
                    if qa['Answer'] == original_key:
                        new_answer_key = chr(65 + new_key)
                
                option_dict = {chr(65 + new_key): value for new_key, (original_key, value) in enumerate(options_items)}
                
                # Build question
                questions_text = qa['Question']
                questions.append({
                    "text": f"{questions_text} \n{options_text}",
                    "options": option_dict,
                })
                
                # Record answer and question type
                answers.append(new_answer_key)
                question_types.append(qa.get('type', 'unknown'))
            break
            
    return questions, answers, question_types

def main():
    parser = argparse.ArgumentParser(description="ThinkAD MMAD Evaluation")
    parser.add_argument("--model_path", type=str, default="/data1/chaoyuhao/models/mmad-3shot-grpo-1000")
    parser.add_argument("--data_path", type=str, default="../../../dataset/MMAD")
    parser.add_argument("--json_path", type=str, default="../../../dataset/MMAD/mmad.json")
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--domain_knowledge_path", type=str, default="../../../dataset/MMAD/domain_knowledge.json")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing multiple images at once")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on")
    args = parser.parse_args()
    
    # 加载模型
    logger.info(f"Loading model from {args.model_path}")
    try:
        if args.device == "auto":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True
            )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        logger.info("Model, processor and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 准备结果保存
    model_name = os.path.basename(os.path.normpath(args.model_path))
    if args.domain_knowledge:
        print("Add domain_knowledge")
        model_name += "_Domain_knowledge"
    if args.similar_template:
        print("Add similar_template")
        model_name += "_Similar_template"
        
    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    os.makedirs("result", exist_ok=True)
    logger.info(f"Results will be saved to {answers_json_path}")
    
    # 加载已有结果（如果存在）
    if os.path.exists(answers_json_path) and not args.reproduce:
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
        existing_images = [a["image"] for a in all_answers_json]
    else:
        all_answers_json = []
        existing_images = []
    
    # 加载领域知识（如果需要）
    all_domain_knowledge = None
    if args.domain_knowledge:
        logger.info(f"Loading domain knowledge from {args.domain_knowledge_path}")
        with open(args.domain_knowledge_path, "r") as file:
            all_domain_knowledge = json.load(file)
    
    # 加载MMAD数据集
    logger.info(f"Loading MMAD dataset from {args.json_path}")
    with open(args.json_path, "r") as file:
        chat_ad = json.load(file)
    
    # Add batch processing
    BSZ = args.batch_size
    logger.info(f"Using batch size: {BSZ}")
    
    # Prepare all data first
    all_data = []
    for image_path in tqdm(chat_ad.keys(), desc="Preparing data"):
        if image_path in existing_images and not args.reproduce:
            continue
        
        try:
            text_gt = chat_ad[image_path]
            rel_image_path = os.path.join(args.data_path, image_path)
            
            # Parse questions without loading images yet
            questions, answers, question_types = parse_conversation(text_gt)
            if not questions or not answers:
                continue
                
            all_data.append({
                "image_path": rel_image_path,
                "original_path": image_path,
                "text_gt": text_gt,
                "questions": questions,
                "answers": answers,
                "question_types": question_types
            })
        except Exception as e:
            logger.error(f"Error preparing {image_path}: {str(e)}")
            continue
    
    # Process in batches
    for i in tqdm(range(0, len(all_data), BSZ), desc="Processing batches"):
        batch_data = all_data[i:i+BSZ]
        batch_prompts = []
        batch_images = []
        batch_questions = []
        
        # Prepare batch inputs
        for data_item in batch_data:
            for q_idx, question in enumerate(data_item["questions"]):
                # Load and resize image
                img_path = data_item["image_path"]
                test_image = Image.open(img_path).convert("RGB")
                width, height = test_image.size
                if width > 1000 or height > 800:
                    ratio = min(1000 / width, 800 / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    test_image = test_image.resize((new_width, new_height), Image.LANCZOS)
                
                # Prepare prompt
                prompt = f"Test image: <|image_pad|>\n\n{question['text']}\n\nFirst output the thinking process in <think> </think> tags and then output the final answer letter in <answer> </answer> tags."
                
                batch_prompts.append(prompt)
                batch_images.append(test_image)
                batch_questions.append((data_item, q_idx))
        
        # Process batch
        inputs = processor(
            text=batch_prompts,
            images=batch_images,
            padding=True,
            padding_side="left", # SFT model
            return_tensors="pt"
        ).to(model.device)
        
        # Generate answers
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True
            )
        
        # Process outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        
        batch_output_text = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Process results
        for idx, response in enumerate(batch_output_text):
            data_item, q_idx = batch_questions[idx]
            question = data_item["questions"][q_idx]
            correct_answer = data_item["answers"][q_idx]
            
            # Extract thinking and answer
            last_think_start = response.rfind("<think>")
            last_think_end = response.rfind("</think>")
            
            if last_think_start != -1 and last_think_end != -1 and last_think_start < last_think_end:
                thinking = response[last_think_start + 7:last_think_end].strip()
            else:
                thinking = ""
            
            last_answer_start = response.rfind("<answer>")
            last_answer_end = response.rfind("</answer>")
            
            if last_answer_start != -1 and last_answer_end != -1 and last_answer_start < last_answer_end:
                answer_text = response[last_answer_start + 8:last_answer_end].strip()
            else:
                answer_text = ""
            
            # Parse answer letter
            answer_letter = extract_answer_letter(answer_text, question['options'])
            
            # Log result
            logger.info(f"Image: {data_item['original_path']}, Question {q_idx+1}")
            logger.info(f"Thinking: {thinking}")
            logger.info(f"Answer: {answer_letter} (Raw: {answer_text})")
            logger.info(f"Correct answer: {correct_answer}")
            logger.info("-" * 50)
            
            # Add to results
            answer_entry = {
                "image": data_item["original_path"],
                "question": question,
                "question_type": data_item["question_types"][q_idx],
                "thinking": thinking,
                "gpt_answer": answer_letter,
                "correct_answer": correct_answer
            }
            all_answers_json.append(answer_entry)
        
        # Save intermediate results
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)
    
    # 计算总体准确率
    logger.info("Calculating overall accuracy...")
    with open(answers_json_path, 'r') as f:
        all_answers_json = json.load(f)
    caculate_accuracy_mmad(answers_json_path)

if __name__ == "__main__":
    main()
