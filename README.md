# MP16-Pro

You can download the images and metadata of MP16-Pro from huggingface: [Jia-py/MP16-Pro](https://huggingface.co/datasets/Jia-py/MP16-Pro/tree/main)

# Data

IM2GPS3K: [images](http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip) | [metadata](https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv)

YFCC4K: [images](http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip) | [metadata](https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv)

# Environment Setting

```bash
# test on cuda12.0
conda create -n g3 python=3.9
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements
```

# Running samples

1. Geo-alignment

You can run `python run_G3.py` to train the model.

2. Geo-diversification

First, you need to build the index file using `python IndexSearch.py`. 

Parameters in IndexSearch.py
- index name --> which model you want to use for embedding
- dataset --> im2gps3k or yfcc4k
- database --> default mp16

Then, you also need to construct index for negative samples by using `python IndexSearchReverse.py`. 

Then, you can run `llm_predict_hf.py` for llama LMM or `llm_predict.py` for GPT-4o or `llm_predict_mis.py` for mistral to generate llm predictions. For GPT-4o you need to set an API key as `export OPENAI_API_KEY="your_api_key_here"`

After that, `running aggregate_llm_predictions.py` to aggregate the predictions.

3. Geo-verification

`python IndexSearch.py --index=g3 --dataset=im2gps3k` to verificate predictions and evaluate.
