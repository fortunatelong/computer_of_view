import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from scipy.stats import ks_2samp
import warnings
import os
import json
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# 通用字体设置
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("使用默认字体")


class DataAnalyzer:
    """数据变量分析类"""

    def __init__(self, target_col='SeriousDlqin2yrs', results_dir='results'):
        self.target_col = target_col
        self.results_dir = results_dir
        self.numerical_features = [
            'age', 'DebtRatio', 'MonthlyIncome', 'NumberRealEstateLoansOrLines',
            'NumberOfOpenCreditLinesAndLoans', 'RevolvingUtilizationOfUnsecuredLines',
            'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]

        # 存储结果
        self.models = {}
        self.model_results = {}
        self.analysis_results = {}

        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'models'), exist_ok=True)

    def load_data(self, file_path):
        """加载数据并删除第一列索引列"""
        data = pd.read_csv(file_path)
        # 删除第一列索引列
        data = data.drop(columns=data.columns[0], errors='ignore')
        return data

    def univariate_analysis(self, data):
        """单变量分析"""
        print("=== 单变量分析 ===")

        # 基本统计信息
        if self.numerical_features:
            num_stats = data[self.numerical_features].describe()
            print("数值型变量统计:")
            print(num_stats)

            # 保存统计结果
            num_stats.to_csv(os.path.join(self.results_dir, 'univariate_statistics.csv'))

        # 目标变量分布
        target_dist = data[self.target_col].value_counts(normalize=True)
        print(f"\n目标变量分布:\n{target_dist}")

        # 可视化目标变量分布
        plt.figure(figsize=(8, 6))
        target_dist.plot(kind='bar')
        plt.title('Target Variable Distribution')
        plt.xlabel('Target Class')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'images', 'target_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 数值变量分布可视化
        n_features = len(self.numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feature in enumerate(self.numerical_features):
            if i < len(axes):
                data[feature].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'images', 'numerical_features_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        self.analysis_results['univariate'] = {
            'numerical_stats': num_stats.to_dict(),
            'target_distribution': target_dist.to_dict()
        }

        return num_stats, target_dist

    def multivariate_analysis(self, data):
        """多变量相关性分析"""
        print("\n=== 多变量相关性分析 ===")

        # 计算相关性矩阵
        correlation_matrix = data[self.numerical_features + [self.target_col]].corr()

        # 相关性热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', annot_kws={'size': 8})
        plt.title('Variable Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'images', 'correlation_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 与目标变量的相关性排序
        target_corr = correlation_matrix[self.target_col].abs().sort_values(ascending=False)
        print("与目标变量的相关性排序:")
        print(target_corr)

        # 保存相关性结果
        correlation_matrix.to_csv(os.path.join(self.results_dir, 'correlation_matrix.csv'))
        target_corr.to_csv(os.path.join(self.results_dir, 'target_correlation.csv'))

        self.analysis_results['multivariate'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'target_correlation': target_corr.to_dict()
        }

        return correlation_matrix, target_corr

    def missing_value_analysis(self, data):
        """缺失值分析"""
        print("\n=== 缺失值分析 ===")

        missing_stats = data.isnull().sum()
        missing_percent = (missing_stats / len(data)) * 100

        missing_df = pd.DataFrame({
            'Missing Count': missing_stats,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)

        print("缺失值统计:")
        print(missing_df[missing_df['Missing Count'] > 0])

        # 可视化缺失值
        plt.figure(figsize=(10, 6))
        missing_df[missing_df['Missing Count'] > 0]['Missing Percentage'].plot(kind='bar')
        plt.title('Missing Values Percentage by Feature')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'images', 'missing_values.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 保存缺失值分析结果
        missing_df.to_csv(os.path.join(self.results_dir, 'missing_values_analysis.csv'))

        self.analysis_results['missing_values'] = missing_df.to_dict()

        return missing_df

    def data_cleaning(self, data):
        """数据清洗 - 只删除缺失值，不进行填充"""
        print("\n=== 数据清洗 ===")
        print(f"原始数据形状: {data.shape}")

        # 删除缺失值
        data_clean = data.dropna()
        print(f"删除缺失值后数据形状: {data_clean.shape}")
        print(f"删除的样本数量: {len(data) - len(data_clean)}")

        # 基本数据质量检查
        print(f"\n数据质量检查:")
        print(f"剩余样本数: {len(data_clean)}")
        print(f"特征数量: {len(data_clean.columns)}")
        print(f"目标变量分布:")
        print(data_clean[self.target_col].value_counts(normalize=True))

        return data_clean

    def prepare_features(self, data):
        """准备特征数据 - 使用原始特征"""
        print("\n=== 准备特征数据 ===")

        # 使用所有数值特征
        available_features = [f for f in self.numerical_features if f in data.columns]
        X = data[available_features]
        y = data[self.target_col]

        print(f"使用特征: {available_features}")
        print(f"特征形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")

        return X, y

    def calculate_ks(self, y_true, y_pred_proba):
        """计算KS值"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        ks_value = max(tpr - fpr)
        return ks_value

    def train_models(self, X, y):
        """训练多个模型"""
        print("\n=== 模型训练 ===")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 定义模型
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        # 训练并评估每个模型
        for name, model in models.items():
            print(f"\n训练 {name}...")
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # 计算指标
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            ks_value = self.calculate_ks(y_test, y_pred_proba)

            # 存储结果
            results = {
                'model': model,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1_score': f1,
                'auc': auc,
                'ks_value': ks_value
            }

            self.models[name] = model
            self.model_results[name] = results

            print(f"{name} - Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            print(f"AUC: {auc:.4f}, KS: {ks_value:.4f}")

        return self.model_results

    def model_evaluation(self, X, y, model_name):
        """模型评估及可视化"""
        if model_name not in self.models:
            print(f"模型 {model_name} 未训练")
            return

        model = self.models[model_name]
        results = self.model_results[model_name]

        # 预测
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # 计算KS值
        ks_value = self.calculate_ks(y, y_pred_proba)

        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'{model_name} - 混淆矩阵')
        axes[0, 0].set_xlabel('预测值')
        axes[0, 0].set_ylabel('真实值')

        # 2. ROC曲线
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc_score = roc_auc_score(y, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('假正例率')
        axes[0, 1].set_ylabel('真正例率')
        axes[0, 1].set_title(f'{model_name} - ROC曲线')
        axes[0, 1].legend()

        # 3. KS曲线
        axes[1, 0].plot(fpr, tpr - fpr, 'r-', label=f'KS = {ks_value:.4f}')
        axes[1, 0].set_xlabel('阈值')
        axes[1, 0].set_ylabel('TPR - FPR')
        axes[1, 0].set_title(f'{model_name} - KS曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. 特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.Series(model.feature_importances_, index=X.columns)
            feature_imp.sort_values().plot(kind='barh', ax=axes[1, 1])
            axes[1, 1].set_title(f'{model_name} - 特征重要性')
        elif hasattr(model, 'coef_'):
            feature_imp = pd.Series(model.coef_[0], index=X.columns)
            feature_imp.sort_values().plot(kind='barh', ax=axes[1, 1])
            axes[1, 1].set_title(f'{model_name} - 系数')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'images', f'{model_name}_evaluation.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 打印详细评估报告
        print(f"\n=== {model_name} 详细评估 ===")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
        print(f"KS值: {ks_value:.4f}")
        print(f"\n分类报告:")
        print(classification_report(y, y_pred))

        return results

    def random_forest_analysis(self, X, model_name='RandomForest'):
        """随机森林模型详细分析"""
        if model_name not in self.models:
            print(f"模型 {model_name} 未训练")
            return

        model = self.models[model_name]

        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} 没有特征重要性属性")
            return

        print(f"\n=== {model_name} 详细分析 ===")

        # 特征重要性可视化
        feature_imp = pd.Series(model.feature_importances_, index=X.columns)
        feature_imp_sorted = feature_imp.sort_values(ascending=True)

        # 特征重要性柱状图
        plt.figure(figsize=(10, 6))
        feature_imp_sorted.plot(kind='barh')
        plt.title(f'{model_name} - 特征重要性排序')
        plt.xlabel('重要性分数')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'images', f'{model_name}_feature_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 特征重要性表格
        importance_df = pd.DataFrame({
            '特征': feature_imp_sorted.index,
            '重要性': feature_imp_sorted.values
        }).sort_values('重要性', ascending=False)

        print(f"{model_name} 特征重要性:")
        print(importance_df)

        # 保存特征重要性结果
        importance_df.to_csv(os.path.join(self.results_dir, f'{model_name}_feature_importance.csv'),
                             index=False, encoding='utf-8-sig')

        # 随机森林其他信息
        print(f"\n{model_name} 其他信息:")
        print(f"树的数量: {model.n_estimators}")
        print(f"最大深度: {model.max_depth if hasattr(model, 'max_depth') else 'Not specified'}")

        return importance_df

    def compare_models(self):
        """模型比较"""
        if not self.model_results:
            print("没有训练好的模型用于比较")
            return

        # 创建比较表格
        comparison_data = []
        for model_name, results in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Precision': results['precision'],
                'Recall': results['recall'],
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1_score'],
                'AUC': results['auc'],
                'KS': results['ks_value']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('Model', inplace=True)

        print("\n=== 模型比较 ===")
        print(comparison_df)

        # 保存比较结果
        comparison_df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'))

        # 可视化比较
        metrics = ['Precision', 'Recall', 'Accuracy', 'F1-Score', 'AUC', 'KS']
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for i, metric in enumerate(metrics):
            if i < len(axes):
                comparison_df[metric].plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric} 比较')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)

        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'images', 'model_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        return comparison_df

    def save_results(self):
        """保存所有分析结果"""
        print("\n=== 保存分析结果 ===")

        # 保存模型结果
        if self.model_results:
            model_results_simple = {}
            for model_name, results in self.model_results.items():
                model_results_simple[model_name] = {
                    'precision': float(results['precision']),
                    'recall': float(results['recall']),
                    'accuracy': float(results['accuracy']),
                    'f1_score': float(results['f1_score']),
                    'auc': float(results['auc']),
                    'ks_value': float(results['ks_value'])
                }

            with open(os.path.join(self.results_dir, 'model_results.json'), 'w') as f:
                json.dump(model_results_simple, f, indent=4, ensure_ascii=False)
            print("模型结果已保存")

        # 保存分析结果
        with open(os.path.join(self.results_dir, 'analysis_results.json'), 'w') as f:
            json.dump(self.analysis_results, f, indent=4, ensure_ascii=False)
        print("分析结果已保存")

        # 保存模型对象
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(self.results_dir, 'models', f'{model_name}.pkl'))
            print(f"模型 {model_name} 已保存")

        # 生成总结报告
        self._generate_summary_report()

    def _generate_summary_report(self):
        """生成总结报告"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_trained': list(self.models.keys()),
            'best_model': None,
            'best_auc': 0
        }

        if self.model_results:
            # 找出AUC最高的模型
            for model_name, results in self.model_results.items():
                if results['auc'] > report['best_auc']:
                    report['best_auc'] = results['auc']
                    report['best_model'] = model_name

        # 保存报告
        with open(os.path.join(self.results_dir, 'summary_report.json'), 'w') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print("总结报告已保存")

        # 打印总结
        print("\n=== 分析总结 ===")
        print(f"最佳模型: {report['best_model']}")
        print(f"最佳AUC: {report['best_auc']:.4f}")
        print(f"所有结果已保存到目录: {self.results_dir}")


def main():
    """主执行函数"""
    analyzer = DataAnalyzer()

    # 加载数据
    data = analyzer.load_data('1.2 train.csv')
    print(f"数据加载完成，形状: {data.shape}")

    # 缺失值分析
    missing_df = analyzer.missing_value_analysis(data)

    # 数据清洗（只删除缺失值）
    data_clean = analyzer.data_cleaning(data)

    # 单变量分析
    analyzer.univariate_analysis(data_clean)

    # 多变量分析
    analyzer.multivariate_analysis(data_clean)

    # 准备特征数据
    X, y = analyzer.prepare_features(data_clean)

    # 训练模型
    analyzer.train_models(X, y)

    # 模型评估
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for model_name in analyzer.models.keys():
        analyzer.model_evaluation(X_test, y_test, model_name)

        # 随机森林特殊分析
        if model_name == 'RandomForest':
            analyzer.random_forest_analysis(X, model_name)

    # 模型比较
    analyzer.compare_models()

    # 保存所有结果
    analyzer.save_results()

    print("\n=== 分析完成 ===")


if __name__ == "__main__":
    main()
