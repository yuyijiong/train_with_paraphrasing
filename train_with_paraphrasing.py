# Encoding: UTF-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
import transformers

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import warnings
import torch
import re


# 读取指定后缀的文件路径
def get_file_paths(dir, suffix, subfolder=True, exclude_suffix=None):
    file_path_list = []
    if subfolder == False:
        for file in os.listdir(dir):
            if file.endswith(suffix):
                file_path_list.append(os.path.join(dir, file))
    else:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(suffix):
                    file_path_list.append(os.path.join(root, file))
    if exclude_suffix != None:
        file_path_list = [file_path for file_path in file_path_list if not file_path.endswith(exclude_suffix)]

    return file_path_list


# 清洗文本，通用
def clean_text_uni_simple(content: str):
    # 删除 \□\■
    content = content.replace(' ', '').replace('□', '').replace('■', '')
    # 替换长空格为短空格
    content = content.replace(' ', ' ').replace('\u3000', ' ')
    # 如果出现两个以上换行符，则最多保留2个
    content = re.sub('[\n\r]{2,}', '\n\n', content)
    # 如果出现两个以上空格，则最多保留2个
    content = re.sub(' {2,}', '  ', content)
    # # -连续出现大于3次，则替换为---
    content = re.sub(r'-{3,}', '---', content)

    return content.strip()


def tokenize_messages_sft(examples, tokenizer, max_length, answer_start_str: str, answer_end_str: str):
    text = examples['text']
    text = clean_text_uni_simple(text)  # 清洗文本

    messages_label = []
    # 将text按照answer_start_str切分
    text_list = text.split(answer_start_str)
    # 除首个元素外，每个元素用answer_end_str切分，只以第一次出现的answer_end_str切分
    for i, content in enumerate(text_list):
        if i == 0:
            messages_label.append({"role": "no", "content": content})
        else:
            content_list = content.split(answer_end_str, 1)
            messages_label.append({"role": "label", "content": content_list[0]})
            messages_label.append({"role": "no", "content": content_list[1]})

    input_ids = []
    labels = []
    for message in messages_label:
        if message['role'] == 'label':
            # 如果答案只包含空格或者换行符，则丢弃该样本
            if message['content'].strip() == '':
                warnings.warn('答案只包含空格或者换行符，丢弃该样本')
                return {'input_ids': None, 'labels': None}
            # 代表答案部分，作为labels
            content = answer_start_str + message['content'] + answer_end_str
            labels.extend([-100] * len(tokenizer.encode(answer_start_str, add_special_tokens=False)))
            labels.extend(tokenizer.encode(message['content'] + answer_end_str, add_special_tokens=False))
        else:
            # 代表问题部分，不作为labels
            content = message['content']
            labels.extend([-100] * len(tokenizer.encode(content, add_special_tokens=False)))

        input_ids.extend(tokenizer.encode(content, add_special_tokens=False))

    if len(input_ids) != len(labels):
        raise ValueError(
            'input_ids和labels长度不一致，input_ids长度：{}，labels长度：{}'.format(len(input_ids), len(labels)))

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    # 如果labels全为-100，则返回None
    if all([label == -100 for label in labels]):
        warnings.warn('labels全为-100，丢弃该样本')
        return {'input_ids': None, 'labels': None}

    return {'input_ids': input_ids, 'labels': labels}

# 微调，做padding，手动加上bos和eos，手动加上labels
def tokenize_sft(examples, tokenizer, max_length, add_special_tokens=True,truncation=True,padding=False,train_on_output=False,**kwargs):

    def set_labels2(input_ids:list,answer_start_str,answer_end_str):
        labels = [-100] * len(input_ids)
        answer_start_ids=tokenizer.encode(answer_start_str)
        answer_end_ids=tokenizer.encode(answer_end_str)

        start_index_list = [i for i in range(len(input_ids) - 1) if
                        input_ids[i:i+len(answer_start_ids)] == answer_start_ids]
        #从index_list中的每个start_index往后搜索，如果遇到 answer_end_ids，就把这部分的labels设为input_ids，答案不包含answer_start_str，但包含answer_end_str
        for start_index in start_index_list:
            # 从start_index往后搜索
            for end_index in range(start_index + len(answer_start_ids), len(input_ids)):
                #如果找到了answer_end_ids，则把input_ids[start_index+len(answer_start_ids):end_index+len(answer_end_ids)]的labels设为input_ids[start_index+len(answer_start_ids):end_index+len(answer_end_ids)]
                if input_ids[end_index:end_index+len(answer_end_ids)]==answer_end_ids:
                    labels[start_index + len(answer_start_ids):end_index + len(answer_end_ids)] = input_ids[start_index + len(answer_start_ids):end_index + len(answer_end_ids)]
                    break


        #如果labels不为-100数量不大于len(answer_end_ids)或全是-100，则报错
        if len([label for label in labels if label!=-100])<=len(answer_end_ids):
            import warnings
            warnings.warn('labels全是-100')
            return None

        return labels


    # 检查examples['text']的每个item是否以bos和eos开头结尾，如果没有则加上
    if add_special_tokens:
        text_items = [tokenizer.bos_token + item if not item.startswith(tokenizer.bos_token) else item for item in
                      examples['text']]
        text_items = [item + tokenizer.eos_token if not item.endswith(tokenizer.eos_token) else item for item in
                      text_items]
    else:
        text_items = examples['text']

    # 此处返回列表，而不用返回tensor。而且通常不padding
    output = tokenizer(text_items, truncation=truncation, add_special_tokens=False,
                           max_length=max_length, padding=padding, return_tensors=None, return_attention_mask=True)

    if train_on_output:
        # 判断output['input_ids']是一重列表还是二重列表
        if isinstance(output['input_ids'][0], list):
            output['labels'] = [set_labels2(input_ids,**kwargs) for input_ids in output['input_ids']]
        else:
            output['labels'] = set_labels2(output['input_ids'],**kwargs)

    else:
        output['labels'] = output['input_ids'].copy()

    return output

def context_qa_prompt(tokenizer, context, questions: list, answers: list):
    messages = []
    for i in range(len(questions)):
        if i == 0:
            messages.append({"role": "user", "content": context + "/n" + questions[i]})
            messages.append({"role": "assistant", "content": answers[i]})
        else:
            messages.append({"role": "user", "content": questions[i]})
            messages.append({"role": "assistant", "content": answers[i]})

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return {"text": prompt}


def load_json_datasets(json_dir, tokenizer):
    # 读取所有json文件或jsonl文件
    path_list = get_file_paths(json_dir, '.json', subfolder=False) + get_file_paths(json_dir, '.jsonl', subfolder=False)

    print('path_list:', path_list)
    dataset_list = []
    for path in tqdm(path_list, desc='加载数据集', total=len(path_list)):
        dataset = Dataset.from_json(path)
        # 由context-q-a形式形成完整prompt，记为text。
        if 'context' in dataset.column_names and 'questions' in dataset.column_names and 'answers' in dataset.column_names:
            dataset = dataset.map(lambda x: context_qa_prompt(tokenizer, x['context'], x['questions'], x['answers']),
                                  batched=False, num_proc=16)
        # 由messages形成完整prompt，记为text。
        elif "messages" in dataset.column_names:
            dataset = dataset.map(lambda x: {
                "text": tokenizer.apply_chat_template(x['messages'], add_generation_prompt=False, tokenize=False)},
                                  batched=False, num_proc=16)
        else:
            raise Exception('数据集中没有context,questions,answers列或messages列')

        # 只保留text列
        dataset = dataset.select_columns(['text'])

        dataset_list.append(dataset)

    # 合并所有数据集
    concatenated_dataset = concatenate_datasets(dataset_list)

    # 打印训练集和验证集长度
    print('训练集样本数', len(concatenated_dataset))
    # 获取数据集每个样本的text的平均长度
    len_list = list(map(len, concatenated_dataset['text']))
    print('数据集每个样本的text的平均长度', sum(len_list) / len(len_list))
    print('数据集的text的最大长度', max(len_list))
    print('数据集的text的最小长度', min(len_list))

    return concatenated_dataset


def main(base_model_path, output_dir, datasedt_dir):
    num_gpus = int(torch.cuda.device_count())
    model_name = base_model_path

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        eval_steps=1,
        report_to="tensorboard",
        logging_strategy='steps',
        logging_steps=10,
        logging_dir=os.path.join(output_dir, 'logs'),
        save_strategy='steps',
        save_steps=100,
        num_train_epochs=1,
        remove_unused_columns=False,

        optim="adamw_torch",
        weight_decay=0,

        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(16 / num_gpus),

        fp16=False,
        bf16=True,
        # deepspeed='ds_config_constant.json',
        auto_find_batch_size=False,
        load_best_model_at_end=False,
        # torch_compile_backend="inductor",

        neftune_noise_alpha=5,
        seed=1,
    )

    model_max_length = 4096 * 8

    # 准备tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                           padding_side='right',
                                                           add_bos_token=False,
                                                           add_eos_token=False,
                                                           model_max_length=model_max_length,
                                                           trust_remote_code=True, )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 读取数据集
    data_dir = datasedt_dir
    raw_dataset = load_json_datasets(data_dir, tokenizer)

    with training_args.main_process_first(desc="tokenizing"):
        # 将数据的每个sample的text转化为token，不进行padding，不需要加上bos和eos，因为数据中已经有了
        lm_dataset = raw_dataset.map(
            lambda x: tokenize_sft(x, tokenizer=tokenizer,
                                                 max_length=model_max_length,
                                                 add_special_tokens=False,
                                                 truncation=True,
                                                 padding=False,
                                                 train_on_output=True,
                                                 answer_start_str='<|im_start|>assistant\n',
                                                 answer_end_str='<|im_end|>',),
            remove_columns=raw_dataset.column_names,
            batched=True,
            num_proc=16,
            desc="Running tokenizer on dataset", )

    # 删除labels为None的样本
    lm_dataset = lm_dataset.filter(lambda x: x['labels'] is not None, num_proc=16)
    # 打乱训练集
    lm_dataset = lm_dataset.shuffle(seed=0)
    # 样本数
    print('样本数', len(lm_dataset))

    # 计算样本的平均长度
    df = lm_dataset.to_pandas()
    df['length'] = df['input_ids'].apply(lambda x: len(x))
    print('样本平均长度（token数）：', df['length'].mean())
    print('样本最大长度（token数）：', df['length'].max())
    #统计labels中不为-100的个数的平均值
    df['labels_num'] = df['labels'].apply(lambda x: len([label for label in x if label != -100]))
    print('labels中不为-100的个数的平均值：', df['labels_num'].mean())
    print('labels中不为-100的个数的最大值：', df['labels_num'].max())
    del df

    # 准备模型
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 加载模型
    if "qwen" in model_name.lower():
        # 加载qwen2模型
        from model_train.modeling_qwen2_training import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto"
                                                 )
    else:
        # 加载llama模型
        from model_train.modeling_llama_training import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto",
                                                 rope_scaling={"type": "dynamic", "factor": 4.0}  # 对rope进行缩放
                                                 )

        print('model.config.rope_scaling:', model.config.rope_scaling)

    # 给模型开启梯度检查点
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model.gradient_checkpointing = True

    # lora训练模型所有linear层
    lora_config = LoraConfig(
        r=16, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM")

    model = get_peft_model(model, lora_config)

    # 确保名字包含lora的参数都是可训练的
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    # 打印模型大小
    try:
        rank = torch.distributed.get_rank()
        if rank == 0:
            model.print_trainable_parameters()
            print("模型大小:", model.get_memory_footprint() / 1024 / 1024 / 1024, 'GB')
    except:
        model.print_trainable_parameters()
        print("模型大小:", model.get_memory_footprint() / 1024 / 1024 / 1024, 'GB')

    # 准备训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                             padding=True,
                                             return_tensors="pt",
                                             pad_to_multiple_of=None),
    )
    model.config.use_cache = False
    trainer.train()

    try:
        rank = torch.distributed.get_rank()
        if rank == 0:
            # 保存base模型的config
            model.base_model.model.config.save_pretrained(training_args.output_dir)
            # 保存lora参数
            trainer.save_model(output_dir=training_args.output_dir)
    except:
        # 保存base模型的config
        model.base_model.model.config.save_pretrained(training_args.output_dir)
        # 保存lora参数
        trainer.save_model(output_dir=training_args.output_dir)

    print('model saved:', training_args.output_dir)


if __name__ == '__main__':
    main(base_model_path="YOUR_PATH/Qwen2-1.5B-Instruct",
         output_dir="YOUR_PATH/Qwen2-1.5B-Instruct-paraph-trained",
         datasedt_dir="DatasetPath")
