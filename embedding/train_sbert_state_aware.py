'''
python train_sbert_state_aware.py -d WikiTable300_StateAware -bone ance -base state_aware
'''
import sys
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="WikiTable300_StateAware", type=str, help="which dataset to use")
parser.add_argument("-bone", "--backbone", default="ance", type=str, help="which backbone to use")
parser.add_argument("-base", "--baseline", default="state_aware", type=str, help="baseline name")
args = parser.parse_args().__dict__

# 生成datetime后缀
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 修改output_name，添加datetime
output_name = "log/sentence_bert/output_" + args["dataset"] + "_" + args["backbone"] + "_" + args["baseline"] + "_" + current_time + ".txt"

sys.stdout = open(output_name,"w")

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib
import logging
import torch
import transformers

transformers.set_seed(42)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = args["dataset"]
data_path = "./datasets/" + dataset

print(f"Loading dataset from: {data_path}")
print(f"Training configuration:")
print(f"  Dataset: {dataset}")
print(f"  Backbone: {args['backbone']}")
print(f"  Baseline: {args['baseline']}")

dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="test")
print(f"Loaded test set: {len(dev_corpus)} docs, {len(dev_queries)} queries")

# 使用训练集的queries.jsonl文件
modify_corpus, modify_queries, modify_qrels = GenericDataLoader(
    data_path,
    query_file="queries.jsonl"   
).load(split="train")
print(f"Loaded train set: {len(modify_corpus)} docs, {len(modify_queries)} queries")

# 选择backbone模型
if args["backbone"] == "ance":
    model_name = "PLMs/msmarco-roberta-base-ance-firstp"
    output_model_path = "runs/msmarco-roberta-base-ance-firstp_" + args["dataset"] + "-" + args["baseline"] + "_" + current_time
elif args["backbone"] == "TASB":
    model_name = "PLMs/msmarco-distilbert-base-tas-b"
    output_model_path = "runs/msmarco-distilbert-base-tas-b_" + args["dataset"] + "-" + args["baseline"] + "_" + current_time
elif args["backbone"] == "condensor":
    model_name = "PLMs/msmarco-bert-co-condensor"
    output_model_path = "runs/msmarco-bert-co-condensor_" + args["dataset"] + "-" + args["baseline"] + "_" + current_time
else:
    model_name = "PLMs/contriever-base-msmarco"
    output_model_path = "runs/contriever-base-msmarco_" + args["dataset"] + "-" + args["baseline"] + "_" + current_time

print(f"Using model: {model_name}")
print(f"Output path: {output_model_path}")

# 增加序列长度以支持state-aware queries（更长的上下文）
word_embedding_model = models.Transformer(model_name, max_seq_length=512)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# 降低batch size以适应更长的序列
retriever = TrainRetriever(model=model, batch_size=16)

print("Preparing training data...")
train_samples = retriever.load_train(modify_corpus, modify_queries, modify_qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
print(f"Training samples: {len(train_samples)}")

train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

# 使用训练集计算recall
ir_evaluator = retriever.load_ir_evaluator(modify_corpus, modify_queries, modify_qrels)

# 确保输出目录存在
os.makedirs(output_model_path, exist_ok=True)

# 训练参数
num_epochs = 20
evaluation_steps = 100
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

print(f"Training configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {retriever.batch_size}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Evaluation steps: {evaluation_steps}")
print(f"  Max sequence length: 512")

print("Starting training...")
retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=output_model_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)  # 使用混合精度训练以节省显存

print(f"Training completed!")
print(f"Model saved to: {output_model_path}")
print(f"Log saved to: {output_name}")

# 关闭日志文件
sys.stdout.close()
sys.stdout = sys.__stdout__

print(f"Training finished! Check logs at: {output_name}")
print(f"Model saved at: {output_model_path}")