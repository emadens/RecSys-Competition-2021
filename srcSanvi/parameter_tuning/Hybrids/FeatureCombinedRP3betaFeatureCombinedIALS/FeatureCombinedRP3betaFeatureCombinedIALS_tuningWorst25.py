from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.Utils.ICM_preprocessing import *
from src.Utils.confidence_scaling import *
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../../in/data_train.csv")
ICM_all = load_ICM("../../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URMs_train = []
URMs_validation = []
ignore_users_list = []

import numpy as np


for k in range(5):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)

    profile_length = np.ediff1d(URM_train.indptr)
    block_size = int(len(profile_length) * 0.25)

    start_pos = 0 * block_size
    end_pos = min(1 * block_size, len(profile_length))
    sorted_users = np.argsort(profile_length)

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]
    sorted_users = np.argsort(profile_length)

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    ignore_users_list.append(sorted_users[users_not_in_group_flag])

evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False,
                                            ignore_users_list=ignore_users_list)

ICMs_combined = []
for URM in URMs_train:
    ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

from src.Hybrid.MergedHybridRecommender import MergedHybridRecommender
from src.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

from bayes_opt import BayesianOptimization

IALS_recommenders = []
rp3betaCBF_recommenders = []

for index in range(len(URMs_train)):
    IALS_recommenders.append(
        FeatureCombinedImplicitALSRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICM_all,
            verbose=True
        )
    )
    IALS_recommenders[index].fit(
        factors=250,
        regularization=0.01,
        use_gpu=False,
        iterations=69,
        num_threads=4,
        confidence_scaling=linear_scaling_confidence,
        **{
            'URM': {"alpha": 50.0},
            'ICM': {"alpha": 44.147654939191355}
        }
    )

    rp3betaCBF_recommenders.append(
        RP3betaCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICMs_combined[index],
            verbose=False
        )
    )

    rp3betaCBF_recommenders[index].fit(
        topK=int(107.71158810874789),
        alpha=0.3620511581509873,
        beta=0.21533144052273637,
        implicit=False
    )

tuning_params = {
    "hybridAlpha": (0, 1)
}

results = []


def BO_func(
        hybridAlpha
):
    recommenders = []

    for index in range(len(URMs_train)):

        recommender = MergedHybridRecommender(
            URM_train=URMs_train[index],
            recommender1=IALS_recommenders[index],
            recommender2=rp3betaCBF_recommenders[index],
            verbose=False
        )

        recommender.fit(
            alpha=hybridAlpha
        )

        recommenders.append(recommender)

    result = evaluator_validation.evaluateRecommender(recommenders)
    results.append(result)
    return sum(result) / len(result)


optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=30,
    n_iter=50,
)

recommender = MergedHybridRecommender(
    URM_train=URM_all,
    recommender1=rp3betaCBF_recommenders[0],
    recommender2=IALS_recommenders[0]
)
recommender.fit()

import json

with open("logs/FeatureCombined" + recommender.RECOMMENDER_NAME + "_worst25_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
