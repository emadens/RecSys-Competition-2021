if __name__ == '__main__':
    from srcSanvi.src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
    from srcSanvi.src.Utils.load_URM import load_URM
    from srcSanvi.src.Utils.load_ICM import load_ICM
    from srcSanvi.src.Utils.ICM_preprocessing import *
    from srcSanvi.src.Utils.write_submission import write_submission

    URM_all = load_URM("input/data_train.csv")
    ICM_all = load_ICM("input/ICM_SUPER.csv")

    ICM_combined = combine(ICM=ICM_all, URM=URM_all)

    rp3betaCombined_recommender = RP3betaCBFRecommender(
        URM_train=URM_all,
        ICM_train=ICM_combined,
        verbose=False
    )

    rp3betaCombined_recommender.fit(
        topK=int(529.1628484087545),
        alpha=0.45304737831676245,
        beta=0.226647894170121,
        implicit=False
    )

    write_submission(recommender=rp3betaCombined_recommender, target_users_path="input/data_target_users_test.csv",
                     out_path='Submission.csv')
