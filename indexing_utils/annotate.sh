# python indexing_utils/annotate.py \
#     --in-fname /home/yjkim/gragfin-data/news_articles\
#     --out-fname /home/yjkim/gragfin-data/news_articles-annotated\
#     --text-column all_text

# python indexing_utils/annotate-v2.py \
#     --in-fname /home/yjkim/gragfin-data/news_articles\
#     --out-fname /home/yjkim/gragfin-data/news_articles-annotated\
#     --text-column all_text \
#     --rule-based \
#     --pos-tagging 


python indexing_utils/annotate-v3.py \
    --in-fname /home/yjkim/gragfin-data/news_articles\
    --out-fname /home/yjkim/gragfin-data/news_articles-annotated\
    --text-column all_text \
    --rule-based
    # --pos-tagging 