if __name__ == '__main__':
    from srcSanvi.src.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
    from srcSanvi.src.Utils.load_URM import load_URM
    from srcSanvi.src.Utils.load_ICM import load_ICM
    from srcSanvi.src.Utils.ICM_preprocessing import *
    from srcSanvi.src.Utils.write_submission import write_submission
    from srcSanvi.src.Utils.confidence_scaling import *

    URM_all = load_URM("input/data_train.csv")
    ICM_all = load_ICM("input/ICM_SUPER.csv")

    ICM_combined = combine(ICM=ICM_all, URM=URM_all)

    IALS_recommender = FeatureCombinedImplicitALSRecommender(
        URM_train=URM_all,
        ICM_train=ICM_all,
        verbose=True
    )

    IALS_recommender.fit(
        factors=int(398.601583855084),
        regularization=0.01,
        use_gpu=False,
        iterations=int(94.22855449116447),
        num_threads=6,
        confidence_scaling=linear_scaling_confidence,
        **{
            'URM': {"alpha": 42.07374324671451},
            'ICM': {"alpha": 41.72067133975204}
        }
    )

    write_submission(recommender=IALS_recommender, target_users_path="input/data_target_users_test.csv",
                     out_path='Submission.csv')
