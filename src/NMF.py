import numpy as np
import pandas as pd

# SUBPROCESS

def readCSVtoMat (review, product):
    pd_review = pd.read_csv(review)
    pd_product = pd.read_csv(product)

    product_category = pd_product['category'].values

    review_userID = pd_review['user_id'].values
    review_product = pd_review['product_id'].values
    review_rating = pd_review['rating'].values

    # ubah product_id jadi category
    for i in range(len(review_product)):
        temp = int(review_product[1:])
        review_product[i] = product_category[temp]

    resMatrix = []
    cols = []
    rows = []
    # bikin matriks dari ratingnya
    for i in range(len(review_userID)):
        if (review_userID[i] not in rows):
            rows = np.append((rows, review_userID[i]))
            baris_nol = np.zeros_like(resMatrix.shape[0])
            resMatrix = np.vstack((resMatrix, baris_nol))
        if (review_product[i] not in cols) :
            cols = np.append((cols, review_product[i]))
            kolom_nol = np.zeros_like(resMatrix.shape[1])
            resMatrix = np.hstack((resMatrix, kolom_nol))
        rowId = np.where(rows == review_userID[i])[0][0]
        colId = np.where(cols == review_product[i])[0][0]
        if resMatrix[rowId][colId] == 0 :
            resMatrix[rowId][colId] = review_rating[i]

    return resMatrix


def printMat (matriks) :
    r, c = matriks.shape
    for i in range (r):
        for j in range (c) :
            print(f"{matriks[i][j]:.4f}", end = " ")
        print()

def transformH (H, W, V) :
    r, c = H.shape
    newH = np.zeros((r, c))

    numer = W.T @ V
    denom = W.T @ W @ H

    for i in range (r):
        for j in range (c):
            newH[i][j] = H[i][j] * numer[i][j] / denom[i][j]
    return newH

def transformW (H, W, V) :
    r, c = W.shape
    newW = np.zeros((r, c))

    numer = V @ H.T
    denom = W @ H @ H.T

    for i in range (r):
        for j in range (c):
            newW[i][j] = W[i][j] * numer[i][j] / denom[i][j]
    return newW    

def frobeniusNorm (matriks) :
    r,c = matriks.shape
    res = 0
    for i in range(r):
        for j in range(c):
            res += matriks[i][j] ** 2
    res = res ** 0.5
    return res

def getUserInput (cols):
    userInput = np.zeros((1, cols))
    validInput = np.zeros((1, cols), dtype=bool)
    for i in range(cols):
        while (True) :
            try:
                valStr = input(f"Masukkan rating terhadap item {i} (masukkan s atau skip untuk melewati)\n")

                if (valStr == "s" or valStr == "skip") :
                    validInput[0][i] = False
                    break

                valFloat = float(valStr)

                if (valFloat >= 0 and valFloat <= 5) :
                    validInput[0][i] = True
                    break
                else:
                    print("Masukan harus berada di rentang 0-5")

            except ValueError:
                print("Masukan harus berupa angka dalam rentang 0-5")
        
        if (validInput[0][i]) :
            userInput[0][i] = valFloat
        
    return userInput, validInput

def printSortedRecommendation(validInput, resMatrix, k) :
    recomendationList = []
    r,c = resMatrix.shape

    for i in range (c):
        if (not validInput[0][i]) :
            recomendationList.append ({
                "id": i,
                "val": resMatrix[r-1][i]
            })
    
    recomendationList = sorted (
        recomendationList,
        key = lambda x : x["val"],
        reverse=True
    )

    print("rekomendasi item untuk user: ")
    for i in range (len(recomendationList)) :
        item = recomendationList[i]
        print(f"{i+1}. {item['id']} -> {item['val']:.4f}")

    # return recomendationList[:k]

# MAIN

demoMatAwal = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4]
])

row, col = demoMatAwal.shape

userInput, validInput = getUserInput(col)
demoMatAwal = np.append(demoMatAwal, userInput, axis=0)

row, col = demoMatAwal.shape
iter = 100
k = 3

W = np.random.rand(row, k)
H = np.random.rand(k, col)

# print("matriks awal adalah :")
# printMat(demoMatAwal)
# print("mat W awal :")
# printMat(W)
# print("mat H awal :")
# printMat(H)

for i in range(iter) :
    W = transformW(H, W, demoMatAwal) # W_baru
    H = transformH(H, W, demoMatAwal) # H_baru

    if (frobeniusNorm(demoMatAwal - W @ H) < 1e-9) :
        break

# print("hasil matriks adalah : ")
resMatrix = W @ H
# printMat(resMatrix)

# print(f"frobenius normnya = {frobeniusNorm(demoMatAwal - W @ H)}")

printSortedRecommendation(validInput, resMatrix, 5)
# print("rekomendasi item untuk user: ")
# for i in range (len(recomendationList)) :
#     item = recomendationList[i]
#     print(f"{i+1}. {item['id']} -> {item['val']:.4f}")