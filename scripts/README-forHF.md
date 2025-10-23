# Install

```
conda create -n $NAME
conda activate $NAME
conda install python=3.10
pip install -e .
pip install datasets bitsandbytes
pip install -U huggingface_hub transformers accelerate
```

# Logs

현재 초기에 document intelligence processing (IDP) 단계에서 LLM을 open source를 이용해서 처리할 때 문제가 생김.

특히 이러한 작업은 매번 반복하게 되어서 caching을 하게 되는데 serving할 때에도 이러한 문제가 생기지 않게 사용하는 doc에 대한 processing은 offline에서 진행하고 이것을 이용하게 하는 방식으로 관리해야 할 것 같다.

다만 현재 상황에서 실험을 할 땐 End-to-End로 진행하다 보니 IDP부터 실행하게 되어 오래걸리게 되는 문제가 발생.

- /home/yjkim/gragfin/graphrag/prompts/index/extract_graph.py


영문 데이터로 수정하여 진행
- ashraq/financial-news-articles

임베딩 단계에서는 현재 길이가 긴 경우 truncation을 적용하고 있음.

