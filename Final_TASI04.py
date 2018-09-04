import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv
from math import sqrt
import matplotlib.pyplot as plt

def mae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction, ground_truth)

data_jan = pd.read_excel('2016_jan.xlsx',sheetname='jan')
data_feb = pd.read_excel('2016_jan.xlsx',sheetname='feb', header=3)
data_mar = pd.read_excel('2016_jan.xlsx',sheetname='maret', header=3)
data_2013=pd.read_excel('Data Pinjam Tahun 2013_V2.xlsx',header=3)
datas = pd.concat([data_jan,data_feb,data_2013,data_mar])

data_2016 = pd.concat([data_jan,data_feb])
all_data=datas[['ID Anggota','Nama Anggota' ,'ID Buku','ID Master Buku','Judul','Aktifitas']]
datas=all_data.dropna(how='any')

event_type = {
   'Borrow': 1,#pinjam
   'Return': 0, #kembalikan
   'Renew': 1# Perpanjangan  
}

datas['score'] = datas['Aktifitas'].apply(lambda x: event_type[x])
data = semua_data
data_train, data_test = cv.train_test_split(datas,test_size = 0.25)

matriks = pd.pivot_table(data_train, values='score', index='Nama Anggota', columns='ID Master Buku',aggfunc=np.sum).fillna(0)
#Item Based
similar_item = pairwise_distances(matriks.T,metric='cosine')
prediksi_item = hitung_prediksi(similar_item,kind='item')
#User Based
similar_user = pairwise_distances(matriks,metric='cosine')
prediksi_user = hitung_prediksi(similar_user,kind='user')
mae_item = mae(prediksi_item,data_test)
mae_user = mae(prediksi_user,data_test)

#MAE ITEM BASED
print "Hasil Akurasi MAE Item Based Pada 5 Tahun : " ,mae_item
print "Hasil Akurasi MAE User Based Pada 5 Tahun : " ,mae_user

fig,ax = plt.subplots()
ax.plot([data],[mae_user])
ax.plot([data],[mae_item])
ax.set_xlabel('Jumlah Data')
ax.set_ylabel('MAE')
ax.set_title('HASIL')
plt.show()
