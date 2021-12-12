import scipy.sparse as sps
from typing import Optional


def recommend(weights, user_id: int, urm_train: sps.csc_matrix, at: Optional[int] = None, remove_seen: bool = True):
    user_profile = urm_train[user_id]

    ranking = user_profile.dot(weights).A.flatten()

    if remove_seen:
        user_profile_start = urm_train.indptr[user_id]
        user_profile_end = urm_train.indptr[user_id + 1]

        seen_items = urm_train.indices[user_profile_start:user_profile_end]

        ranking[seen_items] = -np.inf

    ranking = np.flip(np.argsort(ranking))
    return ranking[:at]


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, weights, at):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in range(URM_test.shape[0]):

        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

        if len(relevant_items) > 0:
            recommended_items = recommend(weights, user_id=user_id, urm_train=URM_test, at=at, remove_seen=False)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender results are: Precision = {}, Recall = {}, MAP = {}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))


if __name__ == '__main__':
    from srcSanvi.src.SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
    from srcSanvi.src.Utils.load_URM import load_URM
    from srcSanvi.src.Utils.load_ICM import load_ICM
    from srcSanvi.src.Utils.ICM_preprocessing import *
    from srcSanvi.src.Utils.write_submission import write_submission

    URM_all = load_URM("input/data_train.csv")
    ICM_all = load_ICM("input/ICM_SUPER.csv")

    ICM_combined = combine(ICM=ICM_all, URM=URM_all)

    SLIM_recommender = MultiThreadSLIM_ElasticNet(
        URM_train=ICM_combined.T,
        verbose=True
    )

    SLIM_recommender.fit(
        alpha=0.00026894910579512645,
        l1_ratio=0.9186,
        topK=int(344.7),
        workers=6
    )

    SLIM_recommender.URM_train = URM_all

    write_submission(recommender=SLIM_recommender, target_users_path="input/data_target_users_test.csv",
                     out_path='Submission.csv')


