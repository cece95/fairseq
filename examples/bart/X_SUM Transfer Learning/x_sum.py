import torch
from fairseq.models.bart import BARTModel

#https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md

bart = BARTModel.from_pretrained('./', checkpoint_file='model.pt')
#bart.cuda()
bart.eval()  # disable dropout (or leave in train mode to finetune)
#bart.half()
count = 1
bsz = 32
for i in range(1,4):
    with open('articles.source') as source, open('x_sum_{}.hypo'.format(i), 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=6, lenpen=1.0, max_len_b=20*i, min_len=10, no_repeat_ngram_size=3)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=6, lenpen=1.0, max_len_b=20*i, min_len=10, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
