TASK=cnn_dm
for SPLIT in train val
do
  for LANG in source target
  do
    sudo python3 -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "data/$SPLIT.$LANG" \
    --outputs "data/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
