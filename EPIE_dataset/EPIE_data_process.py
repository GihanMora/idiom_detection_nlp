import pandas as pd
words_f = open('F:\idiom_scraper\Static_Idioms_Corpus\Static_Idioms_Words.txt','r',encoding='utf-8')
tags_f = open('F:\idiom_scraper\Static_Idioms_Corpus\Static_Idioms_Tags.txt','r',encoding='utf-8')
candidates_f = open('F:\idiom_scraper\Static_Idioms_Corpus\Static_Idioms_Candidates.txt','r',encoding='utf-8')
words = [w.strip() for w in words_f.readlines()]
tags = [t.strip() for t in tags_f.readlines()]
candidates = [c.strip() for c in candidates_f.readlines()]
tags_fixed = []

for each_t in tags:
    fixed_l = []
    tag_l = each_t.split()
    for k in tag_l:
        if(k=='O'):fixed_l.append(0)
        elif(k=='B-IDIOM' or k=='I-IDIOM'):fixed_l.append(1)
    tags_fixed.append(fixed_l)

df = pd.DataFrame()
df['candidate'] = candidates
df['words'] = words
df['tags'] = tags
df['tags_fixed'] = tags_fixed

df.to_csv("EPIE_dataset.csv")