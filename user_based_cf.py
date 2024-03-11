from collections import defaultdict
import pandas as pd

NUM_REVIEWS = 1000000

# read data and instantiate sets (for vector representations instead of sparse matrix)
df = pd.read_csv('Reviews_Part1.csv', usecols=['user_id', 'business_id', 'stars'])
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
ratingsPerUser = defaultdict(set)
allItems = set()

# build sets
for x in range(NUM_REVIEWS):
    user, rating, item = df['user_id'][x], df['stars'][x], df['business_id'][x]
    ratingsPerUser[user].add((item, rating))
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    allItems.add(item)


# compute the cosine similarity between two user vectors
def computeCosine(user1, user2):
    dict_u1 = dict(ratingsPerUser[user1])
    dict_u2 = dict(ratingsPerUser[user2])
    dot_product = sum(dict_u1[item] * dict_u2[item] for item in dict_u1 if item in dict_u2)
    return dot_product / len(allItems) **2

# find the 5 most similar users to a given user
def mostSimilarFast(u):
    similarities = []
    candidateUsers = set()
    for (item, rating) in ratingsPerUser[u]:
        candidateUsers = candidateUsers.union(usersPerItem[item])
    for u2 in candidateUsers:
        if u2 == u: continue
        sim = computeCosine(u, u2)
        similarities.append((sim, u2))
    similarities.sort(reverse=True)
    return similarities[:5]

# create lists and append each entire review to each list
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for d in range(NUM_REVIEWS):
    user, item = df['user_id'][d], df['business_id'][d]
    reviewsPerUser[user].append(df.iloc[d])
    reviewsPerItem[item].append(df.iloc[d])

# use cosine similarity to predict ratings
# commented out code is method 1 - weighted sum of all vectors by similarity
def predictRatingCosine(user, item):
    ratings = []
    similarities = []
    ratings_similarities = []
    for d in reviewsPerItem[item]:
        u2 = d['user_id']
        if u2 == user: continue
    #     ratings.append(d['stars'])
    #     similarities.append(computeCosine(user, u2))
        ratings_similarities.append((d['stars'], computeCosine(user, u2)))
    ratings_similarities.sort(key=lambda x: x[1])
    ratings = 0
    if len(ratings_similarities) >= 3:
        for k in range(3):
            ratings += ratings_similarities[k][0]
        ratings /= 3
    elif len(ratings_similarities) > 0:
        for k in range(len(ratings_similarities)):
            ratings += ratings_similarities[k][0]
        ratings /= len(ratings_similarities)

    return ratings
    # if (sum(similarities) > 0):
    #     weightedRatings = [(x * y) for x, y in zip(ratings, similarities)]
    #     return sum(weightedRatings) / sum(similarities)

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