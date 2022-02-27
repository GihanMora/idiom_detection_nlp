import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#plot the vocabulary
import ast
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.decomposition import PCA #Grab PCA functions
from sklearn.manifold import TSNE
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def tsne_vocab_plot(vocab_df):

    print(vocab_df.columns)
    # X = [ast.literal_eval(i) for i in vocab_df['embedding']]
    X = vocab_df['embedding'].tolist()
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    vocab_df['tsne-2d-one'] = tsne_results[:, 0]
    vocab_df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", len(vocab_df['label'].unique())),
        # palette=sns.color_palette("hls",14),
        # palette=['green','green','green','green','green','red','green','red','red','red','red','red','green','green'],
        # palette=['#C0EB84','#148B0E','#52A22A','#3B780C','#A5E250','#fdc70c','#895207','#BF3804','#DF2902','#EF2101','#ed683c','#f3903f','#40760B','#295115'],
        data=vocab_df,
        legend="full",
        alpha=0.8
    )

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'] + .02, point['y'], str(point['val']))

    emos = vocab_df['label'].unique()
    print(emos)
    vl_df = pd.DataFrame()
    for ee in emos:
        # print(ee)
        l_df = vocab_df[vocab_df['label'] == ee][:3]
        # print(l_df.head())
        vl_df = pd.concat([vl_df, l_df], ignore_index=True)

    # vl_df = pd.concat(emo_df_list, axis=1,ignore_index=True)

    # print(vl_df)
    label_point(vl_df['tsne-2d-one'], vl_df['tsne-2d-two'], vl_df['label'], plt.gca())
    # plt.savefig(r'E:\Projects\DSI_Gihan\Vocabularies\\\vocab_v1.png')

    plt.show()


# print('emo',len(vocab_df))
# print(vocab_df.head())
#
#
# words = vocab_df['token'].tolist()
# embeddings = [ast.literal_eval(i) for i in vocab_df['embedding'].tolist()]
# eight_label = vocab_df['label'].tolist()

# visualize_embs(eight_label,embeddings,words)
