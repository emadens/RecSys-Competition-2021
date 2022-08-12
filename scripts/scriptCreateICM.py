import numpy as np
import pandas as pd
import scipy.sparse as sps
import math


def TFIDF(ICM, nAttributes, inc):  # ICM in coo
    n_items = 18059
    result = list()
    IDFs = list()

    ICM_csr = sps.csr_matrix(ICM)
    ICM_csc = sps.csc_matrix(ICM)

    v_max = 0

    for j in range(nAttributes):
        column = ICM_csc.indices[ICM_csc.indptr[j]:ICM_csc.indptr[j + 1]]
        if len(column):
            IDFs.append(math.log2(18059/len(column)))
        else:
            IDFs.append(0)

    for i in range(n_items):
        row = ICM_csr.indices[ICM_csr.indptr[i]:ICM_csr.indptr[i + 1]]
        if len(row):
            tf = 1/len(row)
            for j in range(nAttributes):
                if j in row:
                    value = tf*IDFs[j]
                    if value > v_max:
                        v_max = value
                    result.append(i)
                    result.append(j + inc)
                    result.append(value)
    result = np.reshape(np.array(result), (int(len(result)/3), 3))
    return result, v_max

def createEpisodesCat(ICM):
    epList = []

    for c, itemId in enumerate(ICM["row"]):
        if len(epList) <= itemId:
            while len(epList) < itemId:
                epList.append(0)
            epList.append(1)
        else:
            epList[itemId] += 1

    li = []
    maxEp = max(epList)
    minEp = min(epList)
    k = 200
    offset = (maxEp - minEp) / k
    for i in epList:
        div = k + 1
        if i == 0:
            li.append(div)  # to the items with no information about episodes we'll assign different bucket ids
            div += 1
        else:
            li.append(int(i / offset))  # num of episodes shrunk to 20 clusters.
    epList = li

    ep_df = pd.DataFrame({
        "itemId": [i for i in range(18059)],
        "epCateg": epList,
        "data": 1
    })
    # print(ep_df)
    return ep_df


if __name__ == '__main__':
    ICM_genre = pd.read_csv("input/data_ICM_genre.csv")
    ICM_subgenre = pd.read_csv("input/data_ICM_subgenre.csv")
    ICM_channel = pd.read_csv("input/data_ICM_channel.csv")
    ICM_event = pd.read_csv("input/data_ICM_event.csv")
    ep_df = createEpisodesCat(ICM_event)

    ICM_genre_coo = sps.coo_matrix((ICM_genre["data"].values, (ICM_genre["row"].values, ICM_genre["col"].values)))
    ICM_subgenre_coo = sps.coo_matrix(
        (ICM_subgenre["data"].values, (ICM_subgenre["row"].values, ICM_subgenre["col"].values)))
    ICM_event_coo = sps.coo_matrix((ep_df["data"].values, (ep_df["itemId"].values, ep_df["epCateg"].values)))
    ICM_channel_coo = sps.coo_matrix(
        (ICM_channel["data"].values, (ICM_channel["row"].values, ICM_channel["col"].values)))

    ICMs = [ICM_genre_coo, ICM_subgenre_coo, ICM_channel_coo, ICM_event_coo]

    att = [8, 113, 213, 200]
    alpha = [1, 0.8, 0.5, 0.3]
    inc = [0, 8, 8+113, 8+113+213]

    ICM_all = list()
    for i in range(4):
        tfidf, v_max = TFIDF(ICMs[i], att[i], inc[i])
        tfidf[:, 2] = tfidf[:, 2] * alpha[i]/v_max
        ICM_all.append(tfidf)

    final = np.vstack((np.vstack((np.vstack((ICM_all[0], ICM_all[1])), ICM_all[2])), ICM_all[3]))

    a = np.asarray(final)
    np.savetxt("foo.csv", a, delimiter=",")


