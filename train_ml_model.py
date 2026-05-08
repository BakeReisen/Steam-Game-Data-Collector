"""
训练在线人数预测模型
基于 Source data.csv 中的完整数据训练模型
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import re
import json

class PlayerCountPredictor:
    def __init__(self):
        """初始化预测器"""
        self.models = {}
        self.feature_importance = {}
        self.training_stats = {}
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从数据中提取特征
        
        Args:
            df: 原始数据
            
        Returns:
            特征DataFrame
        """
        features = pd.DataFrame()
        
        # 1. 游戏年龄(从发行日期计算)
        def calculate_game_age(date_str):
            if pd.isna(date_str) or date_str in ['N/A', 'Coming Soon', 'TBA', '']:
                return -1
            try:
                year_match = re.search(r'(\d{4})', str(date_str))
                if year_match:
                    release_year = int(year_match.group(1))
                    current_year = datetime.now().year
                    return max(0, current_year - release_year)
            except:
                pass
            return -1
        
        features['game_age_years'] = df['发行日期'].apply(calculate_game_age)
        
        # 2. 是否免费游戏
        features['is_free'] = df['格式化价格'].apply(
            lambda x: 1 if pd.notna(x) and str(x).lower() in ['free', '免费'] else 0
        )
        
        # 3. 价格(数值化)
        def extract_price(price_str):
            if pd.isna(price_str) or str(price_str).lower() in ['free', '免费', 'n/a', '']:
                return 0.0
            try:
                # 提取数字
                numbers = re.findall(r'\d+\.?\d*', str(price_str))
                if numbers:
                    return float(numbers[0])
            except:
                pass
            return 0.0
        
        features['price_numeric'] = df['格式化价格'].apply(extract_price)
        
        # 4. 游戏时长特征
        features['playtime_avg'] = pd.to_numeric(df['平均游戏时长(分钟)'], errors='coerce').fillna(0)
        features['playtime_median'] = pd.to_numeric(df['中位数游戏时长(分钟)'], errors='coerce').fillna(0)
        
        # 5. 游戏时长与价格比(性价比指标)
        features['playtime_price_ratio'] = features.apply(
            lambda row: row['playtime_avg'] / max(row['price_numeric'], 1) if row['price_numeric'] > 0 else row['playtime_avg'],
            axis=1
        )
        
        # 6. 在线人数特征(用于互相预测)
        features['current_players'] = pd.to_numeric(df['当前在线人数'], errors='coerce').fillna(0)
        features['peak_24h'] = pd.to_numeric(df['24小时峰值'], errors='coerce').fillna(0)
        features['peak_alltime'] = pd.to_numeric(df['历史最高在线'], errors='coerce').fillna(0)
        
        # 7. 衍生特征
        # 游戏活跃度(当前在线/24小时峰值)
        features['activity_ratio'] = features.apply(
            lambda row: row['current_players'] / row['peak_24h'] if row['peak_24h'] > 0 else 0,
            axis=1
        )
        
        # 历史增长倍数(历史峰值/24小时峰值)
        features['historical_growth'] = features.apply(
            lambda row: row['peak_alltime'] / row['peak_24h'] if row['peak_24h'] > 0 else 0,
            axis=1
        )
        
        return features
    
    def prepare_training_data(self, df: pd.DataFrame, target_field: str):
        """
        准备训练数据
        
        Args:
            df: 特征DataFrame
            target_field: 目标字段名
            
        Returns:
            (X, y) 特征和目标值
        """
        # 过滤掉目标值为0或缺失的记录
        valid_mask = (df[target_field] > 0)
        df_valid = df[valid_mask].copy()
        
        # 根据预测目标选择特征
        if target_field == 'current_players':
            # 预测当前在线:不使用当前在线本身
            feature_cols = ['game_age_years', 'is_free', 'price_numeric', 
                          'playtime_avg', 'playtime_median', 'playtime_price_ratio',
                          'peak_24h', 'peak_alltime', 'historical_growth']
        elif target_field == 'peak_24h':
            # 预测24小时峰值:不使用24小时峰值和衍生特征
            feature_cols = ['game_age_years', 'is_free', 'price_numeric',
                          'playtime_avg', 'playtime_median', 'playtime_price_ratio',
                          'current_players', 'peak_alltime']
        elif target_field == 'peak_alltime':
            # 预测历史峰值:不使用历史峰值和衍生特征
            feature_cols = ['game_age_years', 'is_free', 'price_numeric',
                          'playtime_avg', 'playtime_median', 'playtime_price_ratio',
                          'current_players', 'peak_24h', 'activity_ratio']
        else:
            raise ValueError(f"Unknown target field: {target_field}")
        
        X = df_valid[feature_cols].copy()
        y = df_valid[target_field].copy()
        
        return X, y, feature_cols
    
    def train_model(self, X, y, target_field: str):
        """
        训练模型并评估
        
        Args:
            X: 特征
            y: 目标值
            target_field: 目标字段名
            
        Returns:
            最佳模型
        """
        print(f"\n{'='*80}")
        print(f"训练 {target_field} 预测模型")
        print(f"{'='*80}")
        print(f"训练样本数: {len(X)}")
        print(f"目标值范围: [{y.min():.0f}, {y.max():.0f}]")
        print(f"目标值均值: {y.mean():.0f}")
        print(f"目标值中位数: {y.median():.0f}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 测试多个模型
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'LinearRegression': LinearRegression()
        }
        
        results = {}
        print(f"\n模型评估:")
        print("─"*80)
        
        for name, model in models.items():
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 评估指标
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_mae = -cv_scores.mean()
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mae': cv_mae
            }
            
            print(f"\n{name}:")
            print(f"  训练集 MAE: {train_mae:,.0f} | RMSE: {train_rmse:,.0f} | R²: {train_r2:.4f}")
            print(f"  测试集 MAE: {test_mae:,.0f} | RMSE: {test_rmse:,.0f} | R²: {test_r2:.4f}")
            print(f"  交叉验证 MAE: {cv_mae:,.0f}")
        
        # 选择测试集MAE最小的模型
        best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
        best_model = results[best_model_name]['model']
        
        print(f"\n✅ 最佳模型: {best_model_name}")
        print(f"   测试集 MAE: {results[best_model_name]['test_mae']:,.0f}")
        print(f"   测试集 R²: {results[best_model_name]['test_r2']:.4f}")
        
        # 特征重要性(如果支持)
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = X.columns
            feature_importance = sorted(zip(feature_names, importances), 
                                       key=lambda x: x[1], reverse=True)
            
            print(f"\n特征重要性 (Top 5):")
            for feat, imp in feature_importance[:5]:
                print(f"  {feat}: {imp:.4f}")
            
            self.feature_importance[target_field] = feature_importance
        
        # 保存训练统计
        self.training_stats[target_field] = {
            'model_name': best_model_name,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_mae': float(results[best_model_name]['test_mae']),
            'test_rmse': float(results[best_model_name]['test_rmse']),
            'test_r2': float(results[best_model_name]['test_r2']),
            'cv_mae': float(results[best_model_name]['cv_mae']),
            'target_mean': float(y.mean()),
            'target_median': float(y.median()),
            'target_min': float(y.min()),
            'target_max': float(y.max())
        }
        
        return best_model
    
    def train_all_models(self, data_file: str):
        """
        训练所有预测模型
        
        Args:
            data_file: 数据文件路径
        """
        print("="*80)
        print("在线人数预测模型训练")
        print("="*80)
        
        # 读取数据
        print(f"\n📂 读取数据: {data_file}")
        df = pd.read_csv(data_file, encoding='utf-8-sig')
        print(f"   总记录数: {len(df)}")
        
        # 提取特征
        print(f"\n🔧 提取特征...")
        features_df = self.extract_features(df)
        print(f"   特征维度: {features_df.shape}")
        
        # 训练三个模型
        targets = ['current_players', 'peak_24h', 'peak_alltime']
        
        for target in targets:
            X, y, feature_cols = self.prepare_training_data(features_df, target)
            
            if len(X) < 10:
                print(f"\n⚠️ {target}: 有效样本数太少 ({len(X)}),跳过")
                continue
            
            model = self.train_model(X, y, target)
            self.models[target] = {
                'model': model,
                'feature_cols': feature_cols
            }
        
        # 保存模型和统计信息
        self.save_models()
        
    def save_models(self):
        """保存训练好的模型"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 保存模型文件
        for target, model_info in self.models.items():
            model_path = os.path.join(script_dir, f"model_{target}.pkl")
            joblib.dump(model_info, model_path)
            print(f"\n💾 保存模型: {model_path}")
        
        # 保存训练统计
        stats_path = os.path.join(script_dir, "model_training_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, indent=2, ensure_ascii=False)
        print(f"💾 保存统计: {stats_path}")
        
        # 保存特征重要性
        importance_path = os.path.join(script_dir, "feature_importance.txt")
        with open(importance_path, 'w', encoding='utf-8') as f:
            f.write("特征重要性分析\n")
            f.write("="*80 + "\n\n")
            
            for target, importances in self.feature_importance.items():
                f.write(f"\n{target}:\n")
                f.write("─"*80 + "\n")
                for feat, imp in importances:
                    f.write(f"  {feat:30s} {imp:.6f}\n")
        
        print(f"💾 保存特征重要性: {importance_path}")
        
    def generate_report(self):
        """生成训练报告"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(script_dir, "MODEL_TRAINING_REPORT.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 在线人数预测模型训练报告\n\n")
            f.write(f"**训练时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 模型概览\n\n")
            f.write("| 目标字段 | 模型类型 | 测试集MAE | 测试集RMSE | R² | 训练样本 |\n")
            f.write("|---------|---------|-----------|-----------|-----|----------|\n")
            
            for target, stats in self.training_stats.items():
                target_cn = {
                    'current_players': '当前在线人数',
                    'peak_24h': '24小时峰值',
                    'peak_alltime': '历史最高在线'
                }.get(target, target)
                
                f.write(f"| {target_cn} | {stats['model_name']} | "
                       f"{stats['test_mae']:,.0f} | {stats['test_rmse']:,.0f} | "
                       f"{stats['test_r2']:.4f} | {stats['train_samples']} |\n")
            
            f.write("\n## 📈 详细统计\n\n")
            
            for target, stats in self.training_stats.items():
                target_cn = {
                    'current_players': '当前在线人数',
                    'peak_24h': '24小时峰值',
                    'peak_alltime': '历史最高在线'
                }.get(target, target)
                
                f.write(f"### {target_cn}\n\n")
                f.write(f"**模型类型:** {stats['model_name']}\n\n")
                f.write(f"**数据集:**\n")
                f.write(f"- 训练样本: {stats['train_samples']}\n")
                f.write(f"- 测试样本: {stats['test_samples']}\n\n")
                
                f.write(f"**目标值统计:**\n")
                f.write(f"- 均值: {stats['target_mean']:,.0f}\n")
                f.write(f"- 中位数: {stats['target_median']:,.0f}\n")
                f.write(f"- 范围: [{stats['target_min']:,.0f}, {stats['target_max']:,.0f}]\n\n")
                
                f.write(f"**模型性能:**\n")
                f.write(f"- 测试集平均绝对误差(MAE): {stats['test_mae']:,.0f}\n")
                f.write(f"- 测试集均方根误差(RMSE): {stats['test_rmse']:,.0f}\n")
                f.write(f"- 测试集决定系数(R²): {stats['test_r2']:.4f}\n")
                f.write(f"- 交叉验证MAE: {stats['cv_mae']:,.0f}\n\n")
                
                # 特征重要性
                if target in self.feature_importance:
                    f.write(f"**特征重要性 (Top 10):**\n\n")
                    for feat, imp in self.feature_importance[target][:10]:
                        f.write(f"- {feat}: {imp:.6f}\n")
                    f.write("\n")
            
            f.write("## 💡 使用说明\n\n")
            f.write("模型已保存为以下文件:\n")
            f.write("- `model_current_players.pkl` - 当前在线人数预测模型\n")
            f.write("- `model_peak_24h.pkl` - 24小时峰值预测模型\n")
            f.write("- `model_peak_alltime.pkl` - 历史最高在线预测模型\n\n")
            
            f.write("在 `data_cleaner.py` 中使用:\n")
            f.write("```python\n")
            f.write("import joblib\n")
            f.write("model_info = joblib.load('model_current_players.pkl')\n")
            f.write("model = model_info['model']\n")
            f.write("predictions = model.predict(X)\n")
            f.write("```\n")
        
        print(f"\n📄 生成报告: {report_path}")


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "Source data.csv")
    
    if not os.path.exists(data_file):
        print(f"❌ 错误: 找不到数据文件")
        print(f"   期望路径: {data_file}")
        return
    
    # 创建预测器并训练
    predictor = PlayerCountPredictor()
    predictor.train_all_models(data_file)
    
    # 生成报告
    predictor.generate_report()
    
    print("\n" + "="*80)
    print("✅ 模型训练完成!")
    print("="*80)
    print("\n📁 生成的文件:")
    print("   - model_current_players.pkl")
    print("   - model_peak_24h.pkl")
    print("   - model_peak_alltime.pkl")
    print("   - model_training_stats.json")
    print("   - feature_importance.txt")
    print("   - MODEL_TRAINING_REPORT.md")
    print("\n下一步: 在 data_cleaner.py 中集成这些模型")


if __name__ == "__main__":
    main()
