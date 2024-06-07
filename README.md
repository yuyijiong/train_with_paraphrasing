# Train long-context LLM with "paraphrasing the original text"
##  <center> Abstract </center>
Many models face challenges in long-context tasks, often showing a "lost in the middle" issue. 
To tackle this challenge, we introduce a novel approach called "Paraphrasing the Original Text".
Through a specialized supervised fine-tuning stage that incorporates paraphrasing information into training samples, we improves the model's retrieval capabilities for long-context scenarios.
Our approach is efficient, requiring minimal overhead with fine-tuning needed on just 9k samples with 1 epoch.

* 📄[Paper](https://arxiv.org/abs/2312.11193)
* 📚[Dataset Download](https://huggingface.co/datasets/yuyijiong/Long-Instruction-with-Paraphrasing)

## 💻 Code
* Use QLora method to training the model with our dataset: `train_with_paraphrasing.py`
* Merge lora weights to the original model: `merge_lora.py`

## 🦙 Trained models 
continuously updating...

|model|link|
|---|---|
|llama3-8b-chinese-chat-32k| [link](https://huggingface.co/yuyijiong/llama3-8b-chinese-chat-32k)|
|Qwen-14b-chat-yarn-32k|[link](https://huggingface.co/yuyijiong/Qwen-14b-chat-yarn-32k)|
|Qwen1.5-4b-chat-paraph|[link](https://huggingface.co/yuyijiong/Qwen1.5-4b-chat-paraph)|
