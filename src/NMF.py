import numpy as np
import pandas as pd
from pathlib import Path

# SUBPROCESS

folder_src = Path(__file__).parent
def readCSVtoMat ():
    path_review = folder_src.parent / "data" / "reviews.csv"
    path_product = folder_src.parent / "data" / "products.csv"
    
    pd_review = pd.read_csv(path_review)
    pd_product = pd.read_csv(path_product)

    product_category = pd_product['category'].values

    review_reviewID = pd_review['review_id'].values
    review_userID = pd_review['user_id'].values
    review_product = pd_review['product_id'].values
    review_rating = pd_review['rating'].values

    # ubah product_id jadi category
    for i in range(len(review_product)):
        temp = review_product[i][1:]
        temp = int(temp)
        review_product[i] = product_category[temp-1]

    rows = np.unique(review_userID)
    cols = np.unique(review_product)
    resMatrix = np.zeros((len(rows), len(cols)))

    user_to_idx = {user: i for i, user in enumerate(rows)}
    prod_to_idx = {prod: j for j, prod in enumerate(cols)}
    
    # bikin matriks dari ratingnya
    for i in range(len(review_reviewID)) :
        rowId = user_to_idx[review_userID[i]]
        colId = prod_to_idx[review_product[i]]

        if resMatrix[rowId][colId] == 0 :
            resMatrix[rowId][colId] = review_rating[i]

    return resMatrix, cols


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
    lenCols = len(cols)
    userInput = np.zeros((1, lenCols))
    validInput = np.zeros((1, lenCols), dtype=bool)
    for i in range(lenCols):
        while (True) :
            try:
                valStr = input(f"Masukkan rating (0 - 5) terhadap item "
                               f"{cols[i]} (masukkan s atau skip untuk melewati)\n")

                if (valStr == "s" or valStr == "skip") :
                    validInput[0][i] = False
                    break

                valFloat = float(valStr)

                if (valFloat >= 0 and valFloat <= 5) :
                    validInput[0][i] = True
                    break
                else:
                    print("Masukan harus berada di rentang 0 - 5")

            except ValueError:
                print("Masukan harus berupa angka dalam rentang 0 - 5")
        
        if (validInput[0][i]) :
            userInput[0][i] = valFloat
        
    return userInput, validInput

def printSortedRecommendation(validInput, resMatrix, listItem, k) :
    recomendationList = []
    r,c = resMatrix.shape

    for i in range (c):
        if (not validInput[0][i]) :
            recomendationList.append ({
                "name": listItem[i],
                "val": resMatrix[r-1][i]
            })
    
    recomendationList = sorted (
        recomendationList,
        key = lambda x : x["val"],
        reverse=True
    )

    print("\nRekomendasi Item Untuk User: ")
    for i in range (min(k, len(recomendationList))) :
        item = recomendationList[i]
        print(f"{i+1}. {item['name']} : ({item['val']:.4f} / 5.0)")
    print()

# MAIN

V, listItem = readCSVtoMat()
row, col = V.shape

userInput, validInput = getUserInput(listItem)
V = np.append(V, userInput, axis=0)

row, col = V.shape

iter = 100
k = 5
toleransi = 1e-9

W = np.random.rand(row, k)
H = np.random.rand(k, col)

for i in range(iter) :
    W = transformW(H, W, V) # W_baru
    H = transformH(H, W, V) # H_baru

    if (frobeniusNorm(V - W @ H) < toleransi) :
        break

resMatrix = W @ H

printSortedRecommendation(validInput, resMatrix, listItem, 5)
