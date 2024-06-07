import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    base_model_path = "YOUR_PATH/Qwen2-1.5B-Instruct"
    lora_weight_path = "YOUR_PATH/Qwen2-1.5B-Instruct-paraph-trained"
    merged_model_path = "YOUR_PATH/Qwen2-1.5B-Instruct-paraph-trained-merged"

    base_model=AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype='auto', device_map={"":"cpu"})

    #If need to set the rope scaling (for LLama)
    #base_model.config.rope_scaling={"type": "dynamic", "factor": 4.0}

    lora_model = PeftModel.from_pretrained(base_model, lora_weight_path,device_map={"":"cpu"})

    # merge
    merged_model = lora_model.merge_and_unload()
    # 打印merged_model的类型
    print(type(merged_model))
    # 打印模型大小
    print("模型大小:", merged_model.get_memory_footprint() / 1024 / 1024 / 1024, 'GB')
    # 保存merged_model
    merged_model.save_pretrained(merged_model_path)
    print('merge done', merged_model_path)

    # 保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)
