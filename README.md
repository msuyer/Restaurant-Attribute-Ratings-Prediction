**Restaurant Attiribute Classification and Customer Rating Prediction using Yelp Data**

  Yelp is an online platform that relies on input from its users to provide reviews of local
businesses. The platform features dedicated pages for specific establishments, including
restaurants, where users can post reviews and give ratings ranging from one to five stars for the
products or services offered. People can use Yelp to search for businesses based on their
preferences. The company can use information about user preferences to help optimize their
system and improve the quality of user experience.

  The Yelp dataset can be found on Kaggle and is composed of five distinct datasets. These
datasets represent businesses, reviews, users, check ins, and tips, and are saved as JSON files.
We used the business, reviews, users, and tips datasets for our applications. In the business
dataset, we utilized information regarding a business’ ID, name, star rating, number of reviews,
different attributes (for example if it accepts credit cards, business delivery, if it offers parking,
bike parking, etc.), and business categories such as ‘bubble tea’, ‘mexican’, ‘burgers’, and
‘coffee and tea.’ In the review dataset we used the columns for the review ID, user ID, business
ID, stars given, and the written review provided by the user. Additionally, the dataset quantifies
the number of cool, useful and funny votes on the review. In the user dataset, we used the
columns for the user’s ID, name, and the number of reviews. Finally, in the tip dataset we used
user ID, business ID, and the text of the tip.

  There are many research questions that can be answered with this data, but we decided to
focus on two tasks. Our first task was to predict business attributes by using review and tip
textual information. We chose a multi-label classification approach to this task. Our second task
was to predict how a user would rate a restaurant based on their previous behavior and ratings.
For this task, we chose two main approaches - collaborative filtering and latent factorization. For
both tasks, we chose to only work with businesses within the “Restaurant” and “Food” categories
to narrow the focus of our research and to save computational resources.
