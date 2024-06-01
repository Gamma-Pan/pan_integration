import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

#%%

nfe_df_all = pd.read_csv("~/Downloads/wandb/wandb_export_2024-05-31T17_08_54.804+03_00.csv")
acc_df_all = pd.read_csv("~/Downloads/wandb/wandb_export_2024-05-31T17_08_44.093+03_00.csv")

# %%
all_cols = nfe_df_all.columns
cols = [all_cols[4], all_cols[10], all_cols[16]]
nfe_df = nfe_df_all[cols]
nfe_df = nfe_df.rename(mapper={cols[0]:'tsit_nfe', cols[1]: 'rk4_nfe', cols[2]:'pan_nfe'} , axis=1)

all_cols = acc_df_all.columns
cols = [all_cols[4], all_cols[10], all_cols[16]]
acc_df = acc_df_all[cols]
acc_df = acc_df.rename(mapper={cols[0]:'tsit_acc', cols[1]: 'rk4_acc', cols[2]:'pan_acc'} , axis=1)

df = nfe_df.join(acc_df)
df.loc[-1] = 6*[0]
df.index = df.index + 1
df.sort_index(inplace=True)
#%%
plt.yscale('log')
# plt.xscale('log')
plt.plot(df['tsit_nfe'], df['tsit_acc'], label='tsit5')
plt.plot(df['rk4_nfe'], df['rk4_acc'], label='rk4')
plt.plot(df['pan_nfe'], df['pan_acc'], label='pan')
plt.legend(loc='best')
plt.xlabel("number of $f$ evalutation")
plt.ylabel('validation accuracy')
plt.ylim([0.98,1])
plt.show()


