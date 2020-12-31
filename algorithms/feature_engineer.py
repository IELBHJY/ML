import numpy as np
import pandas as pd


class FeatureEngineer:
    '''
    1规定划分区间的参数，取定长的间隔将特征放入不同的箱子中，这种方法对异常点比较敏感。(等宽)
    2 根据频率划分箱子，会出现特征相同却不在一个箱子中的情况，需要在划分完成后进行微调。（等频）先对特征值进行sort，然后评估分割点，划分或者合并
    3 1R方法：将前面的m个实例放入箱子中如果后面实例放入箱子时，比对当前实例的标签是否与箱子中大部分实例标签相同，如果相同就放入，如果不相同就形成下一个m大小的新箱子，将实例全部放入箱子后，将箱子中大多数实例标签作为箱子的标签，再将标签相同的箱子合并
    4 基于卡方的离散方法：将数值特征的每个不同值看做一个区间对每个相邻的区间计算卡方统计量，如果大就合并，如果不大于阈值就停止。
    5 或者基于熵的离散方法：使用合成或者分裂的方法根据熵计算和阈值判定来JUDGE是合成还是分裂。
    '''

    @staticmethod
    def feature_discretization(data, drop=False):
        data_df = pd.DataFrame(data)
        param = dict()
        columns = data_df.columns
        for col in columns:
            param[col] = data_df[col].mean()
        for col in columns:
            data_df[col] = data_df[col].astype(float)
            data_df["category_" + str(col)] = data_df[col].apply(lambda x: 0 if x <= param[col] else 1)
        cate_columns = data_df.columns
        for col in columns:
            data_df = pd.concat([data_df,
                                 pd.get_dummies(data_df["category_" + str(col)], prefix="category" + str(col),
                                                prefix_sep='_')], axis=1)
        if drop:
            data_df.drop(columns=cate_columns, inplace=True)
        return data_df
