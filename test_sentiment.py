import torch
from transformers import pipeline

print("Torch:", torch.__version__)


sent_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device='mps',
    dtype=None
)

print(sent_analyzer("I really enjoyed learning machine learning."))