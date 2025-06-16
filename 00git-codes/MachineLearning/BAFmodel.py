import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
class model_cal:
    def __init__(self,datapath= 'BAFdata/modeldata2.csv') -> None:
        # 加载数据
        self.data = pd.read_csv(datapath, index_col=0)
        
        # 特征列定义
        self.feature_lis = ['Kow',
                            'trophic level',
                            'Protein (%)',
                            'Concentration in environmental media(ng/L)',
                            'Salinity (‰)',
                            'Temperature(℃)',
                            'groups',
                            'length']
        
        # 提取特征和目标变量
        self.X = self.data[self.feature_lis]
        self.y = self.data['BAF']
        
        # 数据预处理
        self.LNBAF()  # 对数变换
        self.classify_PFAS()  # 分类处理
    
    def LNBAF(self):
        # 对数变换
        self.y = self.y.apply(np.log)
    
    def classify_PFAS(self):
        # 'PFCAs' => 1, 'PFSAs' => 0
        self.X.loc[:, 'groups'] = self.data['groups'].apply(lambda x: 1 if x == 'PFCAs' else 0)
    
    def build_random_forest(self):
        # 定义 StratifiedKFold（分层抽样）进行五折交叉验证
        stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 存储每一折的 R² 和 MSE
        r2_scores = []
        mse_scores = []
        
        # 存储每个样本在验证集时的预测值
        predictions = []

        # 使用 StratifiedKFold 进行五折交叉验证
        for fold, (train_index, test_index) in enumerate(stratified_kf.split(self.X, self.data['groups']), 1):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            # 初始化随机森林模型
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            # 训练模型
            rf_model.fit(X_train, y_train)
            
            # 预测并计算 R² 和 MSE
            y_pred = rf_model.predict(X_test)
            r2 = rf_model.score(X_test, y_test)  # R² score
            mse = mean_squared_error(y_test, y_pred)  # MSE
            
            # 记录每一折的 R² 和 MSE
            r2_scores.append(r2)
            mse_scores.append(mse)
            
            # 保存每个样本的预测结果和对应的真实值（test_index 是验证集的样本索引）
            fold_predictions = pd.DataFrame({
                'sample_index': self.X.iloc[test_index].index,
                'predicted': y_pred,
                'true': y_test
            })
            fold_predictions['fold'] = fold  # 添加当前折数信息
            predictions.append(fold_predictions)  # 将当前折的结果添加到总结果中

        # 将所有折的预测结果合并成一个 DataFrame
        all_predictions = pd.concat(predictions, ignore_index=True)
        
        # 输出 R² 分数
        print(f'R² Scores for each fold: {r2_scores}')
        print(f'Average R² Score: {np.mean(r2_scores)}')
        
        # 输出均方误差
        print(f'Mean Squared Error for each fold: {mse_scores}')
        print(f'Average Mean Squared Error: {np.mean(mse_scores)}')

        # 返回模型和所有预测结果
        return rf_model, all_predictions


    def train_on_full_data(self):
        # 使用整个数据集训练模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X, self.y)
        y_pred = rf_model.predict(self.X)
        r2 = r2_score(self.y, y_pred)  # R² score
        mse = mean_squared_error(self.y, y_pred)  # MSE
        
        # 输出模型评估结果
        print(f'R² Score on Full Data: {r2}')
        print(f'Mean Squared Error on Full Data: {mse}')
        
        # 将预测结果保存到实例变量中
        self.y_pred = y_pred
        self.y_true = self.y
        self.rf_model = rf_model
        # 返回训练好的模型
        return rf_model

    def get_predictions(self):
        """
        返回训练集的真实值和预测值。
        :return: 一个包含真实值和预测值的 DataFrame
        """
        if not hasattr(self, 'y_pred'):  # 确保模型已经训练过
            raise ValueError("模型尚未训练，请先调用 train_on_full_data() 来训练模型。")
        
        # 返回包含真实值和预测值的 DataFrame
        predictions_df = pd.DataFrame({
            'True Values': self.y_true,
            'Predictions': self.y_pred
        })
        
        return predictions_df
    
    def predict_new_data(self, X_new):
        """
        对新的数据进行预测，并返回预测值和95%的置信区间。
        :param X_new: 需要进行预测的新数据 (DataFrame 或 numpy 数组)，
                      其特征应该与训练数据具有相同的列。
        :return: 预测值的 numpy 数组，以及每个预测值的95%置信区间（上限和下限）。
        """
        if not hasattr(self, 'rf_model'):  # 确保模型已经训练过
            raise ValueError("模型尚未训练，请先调用 train_on_full_data() 来训练模型。")
        
        # 确保新数据的特征列与训练集一致
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new[self.feature_lis]
        
        # 获取每棵树的预测结果
        all_tree_predictions = np.array([tree.predict(X_new) for tree in self.rf_model.estimators_]).T
        
        # 计算预测值的均值和标准差
        predictions_mean = all_tree_predictions.mean(axis=1)  # 预测值的均值
        predictions_std = all_tree_predictions.std(axis=1)   # 预测值的标准差
        
        # 计算95%置信区间（假设预测值近似正态分布，1.96倍标准差表示95%置信区间）
        conf_interval_upper = predictions_mean + 1.96 * predictions_std
        conf_interval_lower = predictions_mean - 1.96 * predictions_std
        
        return predictions_mean, conf_interval_lower, conf_interval_upper

    def batch_predict(self, df: pd.DataFrame):
        """
        批量预测新的数据，输入原始数据集，输出带有预测结果和置信区间的新数据集。
        :param df: 输入的原始 DataFrame，列名为 ['Kow', 'TL', 'Protein', 'Con', 'salinity', 'T_C', 'Groups', 'length']
        :return: 返回一个包含预测结果和95%置信区间的新 DataFrame
        """
        # 将新的 DataFrame 转换为符合训练数据格式
        transformed_df = self.trans_df(df)
        
        # 获取每棵树的预测结果
        from tqdm import tqdm
        all_tree_predictions = np.array([tree.predict(transformed_df) for tree in tqdm(self.rf_model.estimators_, desc="Predicting with trees")]).T
        # all_tree_predictions = np.array([tree.predict(transformed_df) for tree in self.rf_model.estimators_]).T
        
        # 计算预测值的均值和标准差
        predictions_mean = all_tree_predictions.mean(axis=1)
        predictions_std = all_tree_predictions.std(axis=1)
        
        # 计算95%置信区间
        conf_interval_upper = predictions_mean + 1.96 * predictions_std
        conf_interval_lower = predictions_mean - 1.96 * predictions_std
        
        # 将预测结果和置信区间添加到原始 DataFrame 中
        df['pre_LNBAF'] = predictions_mean
        df['conf_interval_lower'] = conf_interval_lower
        df['conf_interval_upper'] = conf_interval_upper
        
        return df
    
    def trans_df(self, df: pd.DataFrame):
        """
        将输入的 DataFrame 转换为符合训练数据格式的 DataFrame。
        :param df: 输入的 DataFrame，列名为 ['Kow', 'TL', 'Protein', 'Con', 'salinity', 'T_C', 'Groups', 'length']
        :return: 转换后的 DataFrame
        """
        ori_lis = [
            'Kow',
            'TL',
            'Protein',
            'Con',
            'salinity',
            'T_C',
            'Groups',
            'Length'
        ]
        _df = df[ori_lis]
        _df.loc[:, 'Groups'] = _df['Groups'].apply(lambda x: 1 if x == 'PFCAs' else 0)
        _df.columns = self.feature_lis  # 将列名转换为训练数据格式
        return _df
    def run_multiple_times_and_calculate_statistics(self, df: pd.DataFrame, n_runs=100, confidence_level=0.95):
        """
        运行模型多次，并计算预测结果的统计特性（均值、标准差、置信区间）。
        :param df: 输入的原始 DataFrame，列名为 ['Kow', 'TL', 'Protein', 'Con', 'salinity', 'T_C', 'Groups', 'length']
        :param n_runs: 运行的次数（默认100次）
        :param confidence_level: 置信区间的水平，默认95%
        :return: 一个包含预测结果均值、置信区间上下限的新 DataFrame
        """
        # 将输入的 DataFrame 转换为模型可以接受的格式
        transformed_df = self.trans_df(df)
        
        all_predictions = []

        # 运行n次训练和预测
        from tqdm import tqdm
        for _ in tqdm(range(n_runs)):
            # 训练模型
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(self.X, self.y)
            
            # 预测
            y_pred = rf_model.predict(transformed_df)
            all_predictions.append(y_pred)
        
        # 将预测结果转换为 numpy 数组，形状为 (n_runs, n_samples)
        all_predictions = np.array(all_predictions)
        
        # 计算每个样本的预测均值和标准差
        mean_predictions = all_predictions.mean(axis=0)
        std_predictions = all_predictions.std(axis=0)
        
        # 计算置信区间的z值
        z_value = 1.96  # 对于95%置信区间，z值大约是1.96
        
        # 计算置信区间：均值 ± z * 标准差
        conf_interval_lower = mean_predictions - z_value * std_predictions
        conf_interval_upper = mean_predictions + z_value * std_predictions
        
        # 将计算结果添加到原始 DataFrame 中
        df['pre_LNBAF'] = mean_predictions
        df['conf_interval_lower'] = conf_interval_lower
        df['conf_interval_upper'] = conf_interval_upper
        
        return df

    def calculate_shap_values(self):
        """
        计算特征的 SHAP 值，并返回每个特征的重要性以及 SHAP 值可视化结果。
        :return: 包含每个特征 SHAP 值的 DataFrame 和 SHAP 可视化图
        """
        import shap
        if not hasattr(self, 'rf_model'):  # 确保模型已经训练过
            raise ValueError("模型尚未训练，请先调用 train_on_full_data() 来训练模型。")
        
        # 创建 SHAP 的解释器
        explainer = shap.TreeExplainer(self.rf_model)
        
        # 计算 SHAP 值
        shap_values = explainer.shap_values(self.X)
        
        # 将 SHAP 值存储到 DataFrame 中
        shap_df = pd.DataFrame(shap_values, columns=self.feature_lis)
        
        # 计算平均绝对 SHAP 值作为特征的重要性
        feature_importance = shap_df.abs().mean(axis=0).sort_values(ascending=False)
        
        # 输出特征重要性
        print("Feature Importance (based on SHAP values):")
        print(feature_importance)
        
        # 可视化 SHAP 值（总体重要性条形图）
        self.feature_name_mapping = {
            'Kow': 'Kow',
            'trophic level': 'Trophic Level',
            'Protein (%)': 'Protein Content (%)',
            'Concentration in environmental media(ng/L)': 'Concentration (ng/L)',
            'Salinity (‰)': 'Salinity (‰)',
            'Temperature(℃)': 'Temperature (℃)',
            'groups': 'PFAS Group (PFCAs=1, PFSAs=0)',
            'length': 'Length'
        }
        X_mapped = self.X.rename(columns=self.feature_name_mapping)
        shap.summary_plot(shap_values, X_mapped, plot_type="bar")
        
        # 可视化 SHAP 值（每个样本的解释）
        shap.summary_plot(shap_values, X_mapped)
        
        return shap_df, feature_importance

    def train_and_validate_multiple_seeds(self, n_seeds=10):
        """
        使用不同的随机种子多次训练和验证模型。
        :param n_seeds: 随机种子数量，表示运行次数（默认10次）。
        :return: 返回每次运行的 R^2 和 MSE 统计数据。
        """
        # 存储每次运行的 R² 和 MSE
        all_r2_scores = []
        all_mse_scores = []
        from tqdm import tqdm
        for seed in tqdm(range(n_seeds)):
            # print(f"Running with random seed: {seed}")
            # 初始化随机森林模型
            rf_model = RandomForestRegressor(n_estimators=100, random_state=seed)

            r2_scores = []
            mse_scores = []

            for _ in range(5):  # 五次随机抽样
                # 随机划分训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=seed,shuffle=True
                )

                # 训练模型
                rf_model.fit(X_train, y_train)

                # 预测并计算 R² 和 MSE
                y_pred = rf_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                r2_scores.append(r2)
                mse_scores.append(mse)

            # 保存当前种子的结果
            all_r2_scores.append(np.mean(r2_scores))
            all_mse_scores.append(np.mean(mse_scores))

            # print(f"Seed {seed}: Average R² = {np.mean(r2_scores)}, Average MSE = {np.mean(mse_scores)}")

        # 计算总体平均值和标准差
        avg_r2 = np.mean(all_r2_scores)
        avg_mse = np.mean(all_mse_scores)
        std_r2 = np.std(all_r2_scores)
        std_mse = np.std(all_mse_scores)

        print("\nOverall Statistics:")
        print(f"R²: Mean = {avg_r2}, Std = {std_r2}")
        print(f"MSE: Mean = {avg_mse}, Std = {std_mse}")

        return {
            'r2_scores': all_r2_scores,
            'mse_scores': all_mse_scores,
            'r2_mean': avg_r2,
            'r2_std': std_r2,
            'mse_mean': avg_mse,
            'mse_std': std_mse
        }

    def train_and_validate_single_split(self,chem_cls_type = 'Abbreviation'):
        """
        使用单次的分层抽样训练并验证模型，返回所有有用的结果。
        :return: 包含模型训练和验证结果的字典
        """
        # 使用 train_test_split 进行一次数据集划分，进行 80% 训练和 20% 测试
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.data['Abbreviation'], random_state=42,
        )
        
        _X_train, _X_test, _y_train, _y_test = train_test_split(
            self.data[chem_cls_type], self.y, test_size=0.2, stratify=self.data['Abbreviation'], random_state=42
        )
        # 初始化随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # 训练模型
        rf_model.fit(X_train, y_train)
        
        # 预测并计算 R² 和 MSE
        y_pred = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)  # R² score
        mse = mean_squared_error(y_test, y_pred)  # MSE
        y_train_pre = rf_model.predict(X_train)
        # 将所有有用的结果保存到字典中
        results = {
            'r2_score': r2,
            'mse': mse,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_train_pre': y_train_pre,
            'y_test': y_test,
            'y_pred': y_pred,
            'rf_model': rf_model,
            'chem_cls_train':_X_train,
            'chem_cls_test':_X_test
        }
        
        # 输出结果
        print(f'R² Score: {r2}')
        print(f'Mean Squared Error: {mse}')
        
        # 返回包含所有结果的字典
        return results