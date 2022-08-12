# RecSys-Competition-2021

## Challenge Description
### Goal
The application domain is TV programs recommendation. The datasets we provide contains both interactions between users and TV shows, as well as features related to the shows. The main goal of the competition is to discover which items (TV shows) a user will interact with.
Each TV show (for instance, "The Big Bang Theory") can be composed by several episodes (for instance, episode 5, season 3). The goal of the recommender system is not recommend a specific episode, but to recommend the TV show.

### Description
The datasets includes around 6.2M interactions, 13k users, 18k items (TV shows) and four feature categories: 8 genres, 213 channels, 113 subgenres and 358k events (episode ids).
The training-test split is done via random holdout, 85% training, 15% test.
The goal is to recommend a list of 10 potentially relevant items for each user. MAP@10 is used for evaluation. You can use any kind of recommender algorithm you wish e.g., collaborative-filtering, content-based, hybrid, etc. written in Python.

### Solution
We managed to use the combination of 5 different models for getting to the result, the strongest model among all resulted to be the SLIM.
