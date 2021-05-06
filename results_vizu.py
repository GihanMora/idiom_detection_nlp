import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

data = {'y_Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }

# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15,6))
axes_list = []
details = [['Experiment 1', [[13456,187],[111,1396]]],
           ['Experiment 2',[[22857,196],[509,1558]]],
           ['Experiment 3',[[120776,218],[117,10259]]],
           ['Experiment 4',[[137562,459],[523,11636]]],
           ['Experiment 5',[[7951,136],[213,540]]],
           ['Experiment 6',[[39716,643],[1648,2193]]],
           ['Experiment 7',[[39665,694],[1738,2103]]]]

for row in axes:
        for col in row:
                # ax1 = col
                axes_list.append(col)
                # g1 = sns.heatmap([[400, 441], [611, 302]], cmap="YlGnBu", cbar=False, ax=ax1)
                # g1.set_ylabel('')
                # g1.set_xlabel('')
                # g1.set_title('HI')
axes_list = axes_list[:-1]
for each_axes,det in zip(axes_list,details):
        g1 = sns.heatmap(det[1], annot=True,fmt="d",cmap="Blues", cbar=False, ax=each_axes)
        g1.set_ylabel('True Label')
        g1.set_xlabel('Predicted Label')
        g1.set_title(det[0])

fig.delaxes(axes[1,3])
fig.delaxes(axes[1,2])
fig.delaxes(axes[1,4])
fig.subplots_adjust(wspace=1)
   # print(col)

plt.savefig('conf.png')
plt.show()
# sn.heatmap([[100,241],[211,302]], annot=True,)
# sn.heatmap([[400,441],[611,302]], annot=True,)
# plt.show()
# axx = sn.heatmap([[100,241],[211,302]], annot=True,)
# axx.set_title('HI')
# axx1 = sn.heatmap([[100,241],[211,302]], annot=True,)
# axx1.set_title('HI')
# plt.show()

#
# for cls, ax in zip(['A','B','C','D','E'], axes.flatten()):
#         fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
#         axx1 = sn.heatmap([[100, 241], [211, 302]], annot=True, )
#         axx1.set_title('HI')
#         ax.title.set_text(type(cls).__name__)
# plt.tight_layout()
# plt.show()

import seaborn
# import matplotlib.pyplot as plt
#
# f,(ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(1,4,sharey=True)
# g1 = sns.heatmap([[400,441],[611,302]],cmap="YlGnBu",cbar=False,ax=ax1)
# g1.set_ylabel('')
# g1.set_xlabel('')
# g2 = sns.heatmap([[400,441],[611,302]],cmap="YlGnBu",cbar=False,ax=ax2)
# g2.set_ylabel('')
# g2.set_xlabel('')
# g3 = sns.heatmap([[1400,441],[611,302]],cmap="YlGnBu",cbar=False,ax=ax3)
# g3.set_ylabel('')
# g3.set_xlabel('')
# g4 = sns.heatmap([[1400,441],[611,302]],cmap="YlGnBu",ax=ax4)
# g4.set_ylabel('')
# g4.set_xlabel('')
#
# plt.show()