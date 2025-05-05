from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration


import torch
from PIL import Image
import re
import json
import os
import logging
from datetime import datetime

log_dir = "inference_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VLM-Inference")

model_path = "/home/chaoyuhao/workbench/MMAD/evaluation/examples/ThinkAD/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-3B-GRPO1/checkpoint-1000"
logger.info(f"Loading model from {model_path}")

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Model, processor and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def run_inference(image_path, question, log_results=True):
    logger.info(f"Running inference on image: {image_path}")
    logger.info(f"Question: {question}")
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        question_template = "<|image_pad|> {Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        formatted_question = question_template.format(Question=question)
        
        inputs = processor(
            text=formatted_question,
            images=image,           
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if not isinstance(response, str):
            response = str(response)
        
        print("\n--- Raw Response ---")
        print(response)
        print("--------------------\n")
        
        last_think_start = response.rfind("<think>")
        last_think_end = response.rfind("</think>")
        
        if last_think_start != -1 and last_think_end != -1 and last_think_start < last_think_end:
            thinking = response[last_think_start + 7:last_think_end].strip()
        else:
            thinking = ""
        
        last_answer_start = response.rfind("<answer>")
        last_answer_end = response.rfind("</answer>")
        
        if last_answer_start != -1 and last_answer_end != -1 and last_answer_start < last_answer_end:
            answer = response[last_answer_start + 8:last_answer_end].strip()
        else:
            answer = ""
        
        if log_results:
            print("--- Extracted Thinking ---")
            print(thinking)
            print("\n--- Extracted Answer ---")
            print(answer)
            print("-------------------------")
        
        result = {
            "image_path": image_path,
            "question": question,
            "raw_response": response,
            "thinking": thinking,
            "answer": answer
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e), "image_path": image_path, "question": question}

def batch_inference(image_paths, questions=None, output_file=None):
    results = []
    
    if questions is None or isinstance(questions, str):
        default_question = questions or "Is it a bad object?"
        questions = [default_question] * len(image_paths)
    
    logger.info(f"Starting batch inference on {len(image_paths)} images")
    
    for img_path, question in zip(image_paths, questions):
        result = run_inference(img_path, question, log_results=True)
        results.append(result)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Batch results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    result = run_inference(
        "/home/chaoyuhao/workbench/MMAD/evaluation/examples/ThinkAD/VLM-R1/src/open-r1-multimodal/src/open_r1/test.png",
        "Is there any defect in this product?",
        log_results=True
    )
    