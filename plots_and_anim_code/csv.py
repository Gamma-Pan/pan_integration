import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, ticker

matplotlib.use('TkAgg')
matplotlib.style.use('seaborn-v0_8-colorblind')

#%%

nfe_df_all = pd.read_csv("~/Downloads/wandb/nfe_cifar.csv")
acc_df_all = pd.read_csv("~/Downloads/wandb/acc_cifar.csv")

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
df = df.iloc[:40]

#%%
df[["tsit_acc_sma", "rk4_acc_sma", "pan_acc_sma"]] = df[["tsit_acc", "rk4_acc", "pan_acc"]].rolling(10).mean()

#%%
plt.plot(df['tsit_nfe'], df['tsit_acc_sma'], label='Tsitouras-5')
plt.plot(df['rk4_nfe'], df['rk4_acc_sma'], label='Runge-Kutta-4')
plt.plot(df['pan_nfe'], df['pan_acc_sma'], label='Proposed')
plt.legend(loc='best')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
plt.grid(alpha=0.4, linewidth=1)
plt.xlabel("NFEs")
plt.ylabel('Validatdimsion Set Accuracy')
plt.ylim([0,1])
plt.show()


