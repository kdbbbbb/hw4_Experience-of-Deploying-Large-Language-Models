from transformers import AutoTokenizer, AutoModel
import torch

# 模型路径（请确保模型已经下载并放置在该路径下）
model_path = "/mnt/data/chatglm3-6b"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()

# 提问内容（多个“意思”的语境问题）
question = (
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 "
    "领导：你这就不够意思了。 小明：小意思，小意思。 "
    "领导：你这人真有意思。 小明：其实也没有别的意思。 "
    "领导：那我就不好意思了。 小明：是我不好意思。"
    "请问：以上“意思”分别是什么意思？"
)

# 生成回答
response, _ = model.chat(tokenizer, query=question, history=[])
print("回答：", response)

