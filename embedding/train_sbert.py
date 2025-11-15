'''
python train_sbert.py
'''
import sys
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
# parser.add_argument("-g", "--gpu", default="3", type=str, help="which gpu to use")
parser.add_argument("-d", "--dataset", default="WikiTable300", type=str, help="which dataset to use")
parser.add_argument("-bone", "--backbone", default="ance", type=str, help="which dataset to use")
parser.add_argument("-base", "--baseline", default="TQA", type=str, help="which dataset to use")
args = parser.parse_args().__dict__

# 生成datetime后缀
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 修改output_name，添加datetime
output_name = "log/sentence_bert/output_" + args["dataset"] + "_" + args["backbone"] + "_" + args["baseline"] + "_" + current_time + ".txt"

sys.stdout = open(output_name,"w")
# os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

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
#torch.cuda.set_device(0)
dataset = args["dataset"]
data_path = "./datasets/" + dataset

dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="test")

# train another
# if args["baseline"] == "TQA":
    # modify_corpus, modify_queries, modify_qrels = GenericDataLoader(data_path,query_file="modify_queries_TQA.jsonl").load(split="train")
# else:
    # modify_corpus, modify_queries, modify_qrels = GenericDataLoader(data_path,query_file="new_queries.jsonl").load(split="train")
    
# ---- train set (只用 queries.jsonl) ----
modify_corpus, modify_queries, modify_qrels = GenericDataLoader(
    data_path,
    query_file="queries.jsonl"   
).load(split="train")

if args["backbone"] == "ance":
    model_name = "PLMs/msmarco-roberta-base-ance-firstp"
    output_name = "runs/msmarco-roberta-base-ance-firstp" + args["dataset"] + "-" + args["baseline"] + "_" + current_time
elif args["backbone"] == "TASB":
    model_name = "PLMs/msmarco-distilbert-base-tas-b"
    output_name = "runs/msmarco-distilbert-base-tas-b" + args["dataset"] + "-" + args["baseline"] + "_" + current_time
elif args["backbone"] == "condensor":
    model_name = "PLMs/msmarco-bert-co-condensor"
    output_name = "runs/msmarco-bert-co-condensor" + args["dataset"] + "-" + args["baseline"] + "_" + current_time
else:
    model_name = "PLMs/contriever-base-msmarco"
    output_name = "runs/contriever-base-msmarco" + args["dataset"] + "-" + args["baseline"] + "_" + current_time
    
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device=device) #device="cuda:0"

retriever = TrainRetriever(model=model, batch_size=32)

train_samples = retriever.load_train(modify_corpus, modify_queries, modify_qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

# ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels) # 计算测试集recall
ir_evaluator = retriever.load_ir_evaluator(modify_corpus, modify_queries, modify_qrels) # 用训练集训练模型 计算recall
model_save_path = output_name
# model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "{}-{}".format("PLMs/msmarco-distilbert-ance-v3-new", dataset))
os.makedirs(model_save_path, exist_ok=True)


num_epochs = 20
evaluation_steps = 100
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=output_name,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                
                use_amp=False)


