from collections import defaultdict
import pandas as pd

NUM_REVIEWS = 1000000

# read data and instantiate sets (for vector representations instead of sparse matrix)
df = pd.read_csv('Reviews_Part1.csv', usecols=['user_id', 'business_id', 'stars'])
df2 = pd.read_csv('business.csv', usecols=['business_id', 'name'])
print(df.columns)
print(df2.columns)
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
ratingsPerItem = defaultdict(set)
allUsers = set()
itemNames = {}

# build sets
for x in range(NUM_REVIEWS):
    user, rating, item = df['user_id'][x], df['stars'][x], df['business_id'][x]
    ratingsPerItem[item].add((user, rating))
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    allUsers.add(user)
    if item not in itemNames:
        itemNames[item] = df2.loc[df2['business_id'] == item]['name'] #

# compute the cosine similarity between two item vectors
def computeCosine(item1, item2):
    dict_i1 = dict(ratingsPerItem[item1])
    dict_i2 = dict(ratingsPerItem[item2])
    dot_product = sum(dict_i1[user] * dict_i2[user] for user in dict_i1 if user in dict_i2)
    return dot_product / len(allUsers) **2

# find the 5 most similar items to a given itme
def mostSimilarFast(i):
    similarities = []
    usersAndRatings = ratingsPerItem[i]
    candidateItems = set()
    for (user, rating) in ratingsPerItem[i]:
        candidateItems = candidateItems.union(itemsPerUser[user])
    for i2 in candidateItems:
        if i2 == i: continue
        sim = computeCosine(i, i2)
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:5]

# an example of the most similar restaurants to a given restaurant
query = str(df['business_id'][42])
print(itemNames[query])
for x in mostSimilarFast(query):
    print(itemNames[x[1]])


# create lists and append each entire review to each list
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for d in range(NUM_REVIEWS):
    user, item = df['user_id'][d], df['business_id'][d]
    reviewsPerUser[user].append(df.iloc[d])
    reviewsPerItem[item].append(df.iloc[d])

# use cosine similarity to predict ratings
# commented out code is method 2 - average of top three most similar vectors
def predictRatingCosine(user, item):
    ratings = []
    similarities = []
    # ratings_similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['business_id']
        if i2 == item: continue
        ratings.append(d['stars'])
        similarities.append(computeCosine(item, i2))
    #     ratings_similarities.append((d['stars'], computeCosine(item, i2)))
    # ratings_similarities.sort(key=lambda x: x[1])
    # ratings = 0
    # if len(ratings_similarities) >= 3:
    #     for k in range(3):
    #         ratings += ratings_similarities[k][0]
    #     ratings /= 3
    # elif len(ratings_similarities) > 0:
    #     for k in range(len(ratings_similarities)):
    #         ratings += ratings_similarities[k][0]
    #     ratings /= len(ratings_similarities)
    # if ratings > 0:
    #     return ratings
    if (sum(similarities) > 0):
        weightedRatings = [(x * y) for x, y in zip(ratings, similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return 0

# print an example of a rating prediction
u, i = df['user_id'][1], df['business_id'][1]
print(predictRatingCosine(u, i))

# define MSE loss calculation
def MSE(predictions, labels):
    differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
    return sum(differences) / len(differences)

# predict known ratings using user-based cf and calculate loss
cfPredictions = [predictRatingCosine(df['user_id'][x], df['business_id'][x]) for x in range(NUM_REVIEWS)]
labels = df['stars'][:NUM_REVIEWS]
print(MSE(cfPredictions, labels))