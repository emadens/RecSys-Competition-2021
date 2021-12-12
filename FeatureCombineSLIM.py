if __name__ == '__main__':
    from srcSanvi.src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from srcSanvi.src.SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
    from srcSanvi.src.Utils.ICM_preprocessing import *
    from srcSanvi.src.Utils.load_ICM import load_ICM
    from srcSanvi.src.Utils.load_URM import load_URM

    URM_all = load_URM("input/data_train.csv")
    ICM_all = load_ICM("input/ICM_SUPER.csv")

    from srcSanvi.src.Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    URMs_train = []
    URMs_validation = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

    ICMs_combined = []
    for URM in URMs_train:
        ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

    recommenders = []

    tuning_params = {
        "l1_ratio": (0, 1),
        "topK": (100, 500),
        "max_iter": (10, 200)
    }

    results = []


    def BO_func(
            l1_ratio,
            topK,
            max_iter
    ):
        recommenders = []

        for index in range(len(URMs_train)):
            recommenders.append(
                MultiThreadSLIM_ElasticNet(
                    URM_train=ICMs_combined[index].T,
                    verbose=False
                )
            )

            recommenders[index].fit(
                alpha=0.0001,
                l1_ratio=l1_ratio,
                topK=int(topK),
                max_iter=int(max_iter),
                workers=6
            )

            recommenders[index].URM_train = URMs_train[index]

        result = evaluator_validation.evaluateRecommender(recommenders)
        results.append(result)
        return sum(result) / len(result)


    from srcSanvi.bayes_opt import BayesianOptimization

    optimizer = BayesianOptimization(
        f=BO_func,
        pbounds=tuning_params,
        verbose=5,
        random_state=5,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=15
    )

    import json

    recommenders.append(
        MultiThreadSLIM_ElasticNet(
            URM_train=ICMs_combined[0].T,
            verbose=False
        )
    )

    recommenders[0].fit()

    with open("logs/FeatureCombined" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(optimizer.max, json_file)
