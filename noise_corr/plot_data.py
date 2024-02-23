from all_imports import *
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

Datasets = './data/noisecorr_Avg_Combined_rem_1_2_3_4_5_filtt.npy'
data = load(Datasets, allow_pickle=True).item()
noisecorr_IE = data['noisecorr_IE']
noisecorr_EI = data['noisecorr_EI']
synden_IE = data['synden_IE']
synden_EI = data['synden_EI']


fig = figure(figsize=(9,3))

ax = fig.add_subplot(1,3,1)
x,y = noisecorr_IE, synden_IE
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'E-to-I', fontsize=8)

ax = fig.add_subplot(1,3,2)
x,y = noisecorr_EI, synden_EI
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'I-to-E', fontsize=8)

ax = fig.add_subplot(1,3,3)
x = concatenate((noisecorr_IE-nanmean(noisecorr_IE), noisecorr_EI-nanmean(noisecorr_EI)))
y = concatenate((synden_IE-nanmean(synden_IE), -synden_EI+nanmean(synden_EI)))
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'Pooled', fontsize=8)

tight_layout()

import matplotlib.pyplot as plt
plt.show()
##

