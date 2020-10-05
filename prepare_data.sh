for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json bpe/fairseq/encoder.json \
        --vocab-bpe bpe/fairseq/vocab.bpe \
        --inputs /datadrive_b/roberta/wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs /datadrive_b/roberta/wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 1; \
done

fairseq-preprocess \
    --only-source \
    --srcdict bpe/fairseq/dict.txt \
    --trainpref /datadrive_b/roberta/wikitext-103-raw/wiki.train.bpe \
    --validpref /datadrive_b/roberta/wikitext-103-raw/wiki.valid.bpe \
    --testpref /datadrive_b/roberta/wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 1
