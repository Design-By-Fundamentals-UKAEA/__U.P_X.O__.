# P values
sample_1_mw_p = 5.025283719266939e-07
sample_2_mw_p = 0.010601995752140984
sample_3_mw_p = 0.8855567000253202

sample_1_kw_p = 5.017379755942188e-07
sample_2_kw_p = 0.010590133943115863
sample_3_kw_p = 0.8851803711233379

sample_1_ks_p = 3.242848996728745e-06
sample_2_ks_p = 0.08850144045118859
sample_3_ks_p = 0.9286815554296503

X = ["Sample-1", "Sample-2", "Sample-3"]
Y = ["MW", "KW", "KS"]
data_mw = [sample_1_mw_p,
           sample_2_mw_p,
           sample_3_mw_p]

data_kw = [sample_1_kw_p,
           sample_2_kw_p,
           sample_3_kw_p]

data_ks = [sample_1_ks_p,
           sample_2_ks_p,
           sample_3_ks_p]

df = pd.DataFrame(np.c_[data_mw, data_kw, data_ks], index=Y)


df.plot.bar()

df = df.T

sns.barplot(data=df)


plt.ylim(0, 1)
plt.ylabel("$P-value$")
plt.legend(
    labels=["Sample-1 @ tslice: 4", "Sample-2 @ tslice: 6", "Sample-3 @ tslice: 8"]
)
plt.xlabel("Statistical test")
plt.show()

data = []
for test, values in zip(Y, [data_mw, data_kw, data_ks]):
    for i, value in enumerate(values, 1):
        data.append({'test': test,
                     'index': i,
                     'value': value
                     })
df = pd.DataFrame(data)

plt.figure(figsize=(3.5, 3.5))
sns.barplot(x='test', y='value', hue='index', data=df, palette=['blue','orange', 'green'])
plt.legend(
    labels=["Sample-1 @ tslice: 4", "Sample-2 @ tslice: 6", "Sample-3 @ tslice: 8"]
)
