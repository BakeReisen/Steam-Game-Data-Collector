"""
Steam 游戏数据清洗脚本
对 Source data.csv 进行数据预处理:
1. 检测关键字段的缺失值(空值或0)
2. 尝试根据 AppID 爬取缺失数据
3. 删除无法补全的异常记录
"""

import pandas as pd
import requests
import time
import random
from typing import Dict, Optional, Tuple
from bs4 import BeautifulSoup
import re
from datetime import datetime
import joblib
import os
import numpy as np

class SteamDataCleaner:
    def __init__(self):
        """初始化数据清洗器"""
        self.store_api_base = "https://store.steampowered.com/api"
        self.web_api_base = "https://api.steampowered.com"
        self.steamspy_api_base = "https://steamspy.com/api.php"
        
        # 随机User-Agent列表
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        
        # 需要检查的关键字段(中文列名)
        self.critical_fields = {
            '发行日期': 'release_date',
            '格式化价格': 'price',
            '平均游戏时长(分钟)': 'playtime_avg',
            '中位数游戏时长(分钟)': 'playtime_median',
            '当前在线人数': 'current_players',
            '24小时峰值': 'peak_24h',
            '历史最高在线': 'peak_alltime'
        }
        
        # 加载机器学习模型
        self.ml_models = {}
        self._load_ml_models()
    
    def _load_ml_models(self):
        """加载训练好的机器学习模型"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_files = {
            'current_players': 'model_current_players.pkl',
            'peak_24h': 'model_peak_24h.pkl',
            'peak_alltime': 'model_peak_alltime.pkl'
        }
        
        for target, filename in model_files.items():
            model_path = os.path.join(script_dir, filename)
            if os.path.exists(model_path):
                try:
                    self.ml_models[target] = joblib.load(model_path)
                except Exception as e:
                    print(f"⚠️ 无法加载模型 {filename}: {e}")
        
        if self.ml_models:
            print(f"[ML] 成功加载 {len(self.ml_models)} 个机器学习模型")
        else:
            print(f"[警告] 未找到机器学习模型,将使用简单估算")
    
    def _extract_features_for_prediction(self, df_row) -> Dict:
        """
        从数据行提取特征用于预测
        
        Args:
            df_row: DataFrame行
            
        Returns:
            特征字典
        """
        features = {}
        
        # 1. 游戏年龄
        release_date_str = df_row.get('发行日期', 'N/A')
        game_age_years = -1
        if release_date_str and release_date_str not in ['N/A', 'Coming Soon', 'TBA', '']:
            try:
                year_match = re.search(r'(\d{4})', str(release_date_str))
                if year_match:
                    release_year = int(year_match.group(1))
                    current_year = datetime.now().year
                    game_age_years = max(0, current_year - release_year)
            except:
                pass
        features['game_age_years'] = game_age_years
        
        # 2. 是否免费
        price_str = df_row.get('格式化价格', '')
        features['is_free'] = 1 if str(price_str).lower() in ['free', '免费'] else 0
        
        # 3. 价格数值
        price_numeric = 0.0
        if price_str and str(price_str).lower() not in ['free', '免费', 'n/a', '']:
            try:
                numbers = re.findall(r'\d+\.?\d*', str(price_str))
                if numbers:
                    price_numeric = float(numbers[0])
            except:
                pass
        features['price_numeric'] = price_numeric
        
        # 4. 游戏时长
        features['playtime_avg'] = float(df_row.get('平均游戏时长(分钟)', 0) or 0)
        features['playtime_median'] = float(df_row.get('中位数游戏时长(分钟)', 0) or 0)
        
        # 5. 时长价格比
        if price_numeric > 0:
            features['playtime_price_ratio'] = features['playtime_avg'] / price_numeric
        else:
            features['playtime_price_ratio'] = features['playtime_avg']
        
        # 6. 在线人数
        features['current_players'] = float(df_row.get('当前在线人数', 0) or 0)
        features['peak_24h'] = float(df_row.get('24小时峰值', 0) or 0)
        features['peak_alltime'] = float(df_row.get('历史最高在线', 0) or 0)
        
        # 7. 衍生特征
        if features['peak_24h'] > 0:
            features['activity_ratio'] = features['current_players'] / features['peak_24h']
            features['historical_growth'] = features['peak_alltime'] / features['peak_24h']
        else:
            features['activity_ratio'] = 0
            features['historical_growth'] = 0
        
        return features
    
    def is_field_missing(self, value, field_type: str) -> bool:
        """
        判断字段值是否缺失
        
        Args:
            value: 字段值
            field_type: 字段类型 ('release_date', 'price', 'numeric')
            
        Returns:
            True 如果缺失, False 否则
        """
        # 处理 NaN 和 None
        if pd.isna(value) or value is None:
            return True
        
        # 字符串类型检查
        if isinstance(value, str):
            value = value.strip()
            if value == '' or value.lower() == 'n/a' or value.lower() == 'nan':
                return True
            if field_type == 'release_date' and value in ['N/A', 'Coming Soon', 'TBA']:
                return True
            if field_type == 'price' and value in ['N/A', 'Free', '']:
                return False  # Free 是有效值
        
        # 数值类型检查(对于在线人数等字段)
        if field_type in ['current_players', 'peak_24h', 'peak_alltime', 'playtime_avg', 'playtime_median']:
            try:
                num_value = float(value)
                return num_value == 0 or pd.isna(num_value)
            except:
                return True
        
        return False
    
    def get_game_details(self, app_id: int, max_retries: int = 3) -> Optional[Dict]:
        """
        从 Steam API 获取游戏详情
        
        Args:
            app_id: Steam AppID
            max_retries: 最大重试次数
            
        Returns:
            游戏详情字典或 None
        """
        url = f"{self.store_api_base}/appdetails"
        params = {
            "appids": app_id,
            "cc": "cn",
            "l": "schinese"
        }
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': random.choice(self.user_agents),
                    'Accept': 'application/json'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=20)
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 30
                    print(f"      触发速率限制, 等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                app_data = data.get(str(app_id), {})
                if app_data.get('success'):
                    return app_data['data']
                return None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"      获取详情失败: {e}")
                    return None
        
        return None
    
    def get_player_count(self, app_id: int) -> Optional[int]:
        """
        从 SteamSpy 获取当前在线人数 (CCU)
        
        Args:
            app_id: Steam AppID
            
        Returns:
            在线人数或 None
        """
        try:
            steamspy_data = self.get_steamspy_data(app_id)
            if steamspy_data and steamspy_data.get('ccu'):
                return steamspy_data.get('ccu')
            return None
        except Exception as e:
            return None
    
    def get_steamspy_data(self, app_id: int) -> Optional[Dict]:
        """
        从 SteamSpy 获取游戏数据
        
        Args:
            app_id: Steam AppID
            
        Returns:
            SteamSpy 数据字典或 None
        """
        url = self.steamspy_api_base
        params = {
            "request": "appdetails",
            "appid": app_id
        }
        
        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'application/json',
                'Referer': 'https://steamspy.com/'
            }
            response = requests.get(url, params=params, headers=headers, timeout=25)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"      获取 SteamSpy 数据失败: {e}")
            return None
    
    def get_steamcharts_data(self, app_id: int) -> Tuple[Optional[int], Optional[int]]:
        """
        从 SteamCharts 获取峰值数据
        
        Args:
            app_id: Steam AppID
            
        Returns:
            (24小时峰值, 历史峰值) 元组
        """
        url = f"https://steamcharts.com/app/{app_id}"
        
        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 404:
                # 游戏不存在,使用 SteamSpy CCU 作为备用
                steamspy_data = self.get_steamspy_data(app_id)
                if steamspy_data:
                    ccu = steamspy_data.get('ccu', 0)
                    if ccu and ccu > 0:
                        return ccu, None
                return None, None
            
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            peak_24h = None
            peak_alltime = None
            
            # SteamCharts 的数据在表格中
            # 查找表格行,最后一列是峰值
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')
                
                # 遍历表格行,查找峰值数据
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        # 最后一列是峰值(class="right num")
                        peak_cell = cells[-1]
                        if peak_cell.get('class') and 'num' in peak_cell.get('class'):
                            try:
                                peak_text = peak_cell.get_text().strip()
                                # 移除逗号并转换为整数
                                peak_value = int(peak_text.replace(',', ''))
                                
                                # 更新历史最高峰值
                                if peak_alltime is None or peak_value > peak_alltime:
                                    peak_alltime = peak_value
                            except:
                                continue
            
            # 如果找到了历史峰值,使用 SteamSpy CCU 作为24小时峰值
            if peak_alltime:
                steamspy_data = self.get_steamspy_data(app_id)
                if steamspy_data:
                    ccu = steamspy_data.get('ccu', 0)
                    if ccu and ccu > 0:
                        peak_24h = ccu
                
                return peak_24h, peak_alltime
            
            # 如果没有找到数据,返回 None
            return None, None
            
        except Exception as e:
            # 静默失败,避免刷屏
            return None, None
    
    def estimate_player_fields(self, df_row, missing_fields: list) -> Dict:
        """
        使用机器学习模型估算在线人数相关字段
        
        Args:
            df_row: DataFrame 行数据
            missing_fields: 缺失字段列表
            
        Returns:
            估算的数据字典
        """
        估算数据 = {}
        
        # 如果没有加载模型,使用简单规则估算
        if not self.ml_models:
            return self._simple_estimate(df_row, missing_fields)
        
        # 提取特征
        features_dict = self._extract_features_for_prediction(df_row)
        
        # 按优先级进行预测(先预测依赖少的字段)
        # 优先级: peak_24h > current_players, peak_alltime
        
        # 1. 预测 24小时峰值 (如果缺失)
        if 'peak_24h' in missing_fields and 'peak_24h' in self.ml_models:
            try:
                model_info = self.ml_models['peak_24h']
                model = model_info['model']
                feature_cols = model_info['feature_cols']
                
                # 准备特征向量 (使用DataFrame保留列名)
                X = pd.DataFrame([[features_dict.get(col, 0) for col in feature_cols]], 
                                columns=feature_cols)
                
                # 预测
                prediction = model.predict(X)[0]
                prediction = max(0, int(prediction))  # 确保非负
                
                if prediction > 0:
                    估算数据['24小时峰值'] = prediction
                    features_dict['peak_24h'] = prediction  # 更新用于后续预测
            except Exception as e:
                print(f"      ⚠️ 模型预测失败 (peak_24h): {e}")
        
        # 2. 预测 当前在线人数 (如果缺失)
        if 'current_players' in missing_fields and 'current_players' in self.ml_models:
            try:
                model_info = self.ml_models['current_players']
                model = model_info['model']
                feature_cols = model_info['feature_cols']
                
                X = pd.DataFrame([[features_dict.get(col, 0) for col in feature_cols]], 
                                columns=feature_cols)
                prediction = model.predict(X)[0]
                prediction = max(0, int(prediction))
                
                if prediction > 0:
                    估算数据['当前在线人数'] = prediction
                    features_dict['current_players'] = prediction
            except Exception as e:
                print(f"      ⚠️ 模型预测失败 (current_players): {e}")
        
        # 3. 预测 历史最高在线 (如果缺失)
        if 'peak_alltime' in missing_fields and 'peak_alltime' in self.ml_models:
            try:
                # 重新计算衍生特征
                if features_dict['peak_24h'] > 0:
                    features_dict['activity_ratio'] = features_dict['current_players'] / features_dict['peak_24h']
                
                model_info = self.ml_models['peak_alltime']
                model = model_info['model']
                feature_cols = model_info['feature_cols']
                
                X = pd.DataFrame([[features_dict.get(col, 0) for col in feature_cols]], 
                                columns=feature_cols)
                prediction = model.predict(X)[0]
                prediction = max(0, int(prediction))
                
                if prediction > 0:
                    估算数据['历史最高在线'] = prediction
            except Exception as e:
                print(f"      ⚠️ 模型预测失败 (peak_alltime): {e}")
        
        return 估算数据
    
    def _simple_estimate(self, df_row, missing_fields: list) -> Dict:
        """
        简单规则估算(备用方法)
        
        Args:
            df_row: DataFrame行
            missing_fields: 缺失字段列表
            
        Returns:
            估算数据字典
        """
        估算数据 = {}
        
        # 获取发行年龄
        release_date_str = df_row.get('发行日期', 'N/A')
        game_age_years = 0
        if release_date_str and release_date_str not in ['N/A', 'Coming Soon', 'TBA']:
            try:
                year_match = re.search(r'(\d{4})', str(release_date_str))
                if year_match:
                    release_year = int(year_match.group(1))
                    current_year = datetime.now().year
                    game_age_years = max(0, current_year - release_year)
            except:
                pass
        
        # 衰减系数
        if game_age_years <= 2:
            decay_factor = 0.8
            alltime_factor = 1.3
        elif game_age_years <= 5:
            decay_factor = 0.5
            alltime_factor = 2.0
        else:
            decay_factor = 0.2
            alltime_factor = 3.5
        
        # 获取现有数据
        current_players = float(df_row.get('当前在线人数', 0) or 0)
        peak_24h = float(df_row.get('24小时峰值', 0) or 0)
        peak_alltime = float(df_row.get('历史最高在线', 0) or 0)
        
        # 估算
        if 'peak_24h' in missing_fields:
            if current_players > 0:
                估算数据['24小时峰值'] = int(current_players / decay_factor)
                peak_24h = 估算数据['24小时峰值']
            elif peak_alltime > 0:
                估算数据['24小时峰值'] = int(peak_alltime / alltime_factor)
                peak_24h = 估算数据['24小时峰值']
        
        if 'current_players' in missing_fields and peak_24h > 0:
            估算数据['当前在线人数'] = int(peak_24h * decay_factor)
        
        if 'peak_alltime' in missing_fields and peak_24h > 0:
            估算数据['历史最高在线'] = int(peak_24h * alltime_factor)
        
        return 估算数据
    
    def fetch_missing_data(self, app_id: int, missing_fields: list) -> Dict:
        """
        根据缺失字段爬取数据
        
        Args:
            app_id: Steam AppID
            missing_fields: 缺失字段列表
            
        Returns:
            补充的数据字典
        """
        补充数据 = {}
        
        # 判断需要调用哪些API
        需要steam详情 = any(field in ['release_date', 'price'] for field in missing_fields)
        需要steamspy = any(field in ['playtime_avg', 'playtime_median', 'current_players', 'peak_24h'] for field in missing_fields)
        需要steamcharts = 'peak_alltime' in missing_fields or 'peak_24h' in missing_fields
        
        # 获取 Steam 详情
        if 需要steam详情:
            game_details = self.get_game_details(app_id)
            time.sleep(0.5)
            
            if game_details:
                # 发行日期
                if 'release_date' in missing_fields:
                    release_date = game_details.get('release_date', {})
                    date_str = release_date.get('date', 'N/A')
                    if date_str and date_str != 'N/A':
                        补充数据['发行日期'] = date_str
                
                # 价格
                if 'price' in missing_fields:
                    price_overview = game_details.get('price_overview', {})
                    if price_overview:
                        price_formatted = price_overview.get('final_formatted', 'N/A')
                        补充数据['格式化价格'] = price_formatted
                    elif game_details.get('is_free', False):
                        补充数据['格式化价格'] = 'Free'
        
        # 获取 SteamSpy 数据 (游戏时长 + 在线人数 + 峰值)
        if 需要steamspy:
            steamspy_data = self.get_steamspy_data(app_id)
            time.sleep(0.5)
            
            if steamspy_data:
                # 游戏时长
                if 'playtime_avg' in missing_fields:
                    avg_forever = steamspy_data.get('average_forever', 0)
                    if avg_forever and avg_forever > 0:
                        补充数据['平均游戏时长(分钟)'] = avg_forever
                
                if 'playtime_median' in missing_fields:
                    median_forever = steamspy_data.get('median_forever', 0)
                    if median_forever and median_forever > 0:
                        补充数据['中位数游戏时长(分钟)'] = median_forever
                
                # 当前在线人数 (CCU)
                if 'current_players' in missing_fields:
                    ccu = steamspy_data.get('ccu', 0)
                    if ccu and ccu > 0:
                        补充数据['当前在线人数'] = ccu
                
                # 24小时峰值 (使用 CCU 作为估计)
                if 'peak_24h' in missing_fields:
                    ccu = steamspy_data.get('ccu', 0)
                    if ccu and ccu > 0:
                        补充数据['24小时峰值'] = ccu
        
        # 获取 SteamCharts 数据 (峰值)
        if 需要steamcharts:
            peak_24h, peak_alltime = self.get_steamcharts_data(app_id)
            time.sleep(0.5)
            
            # 优先使用 SteamCharts 的数据
            if 'peak_24h' in missing_fields and peak_24h and '24小时峰值' not in 补充数据:
                补充数据['24小时峰值'] = peak_24h
            
            if 'peak_alltime' in missing_fields and peak_alltime:
                补充数据['历史最高在线'] = peak_alltime
        
        return 补充数据
    
    def clean_data(self, input_file: str, output_file: str = None):
        """
        清洗数据主函数
        
        Args:
            input_file: 输入 CSV 文件路径
            output_file: 输出 CSV 文件路径 (默认在原文件名后加 _cleaned)
        """
        print("=" * 80)
        print("Steam 游戏数据清洗程序")
        print("=" * 80)
        
        # 读取数据
        print(f"\n[读取] 数据文件: {input_file}")
        try:
            df = pd.read_csv(input_file, encoding='utf-8-sig')
            print(f"[成功] 读取 {len(df)} 条记录, {len(df.columns)} 个字段")
        except Exception as e:
            print(f"[错误] 读取文件失败: {e}")
            return
        
        # 检查必需列是否存在
        required_columns = ['AppID'] + list(self.critical_fields.keys())
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"[错误] 缺少必需的列: {missing_columns}")
            return
        
        # 统计初始状态
        print(f"\n[分析] 数据质量分析:")
        records_with_missing = 0
        field_missing_count = {field: 0 for field in self.critical_fields.keys()}
        
        for idx, row in df.iterrows():
            has_missing = False
            for field_cn, field_type in self.critical_fields.items():
                if self.is_field_missing(row[field_cn], field_type):
                    field_missing_count[field_cn] += 1
                    has_missing = True
            if has_missing:
                records_with_missing += 1
        
        print(f"  存在缺失数据的记录: {records_with_missing}/{len(df)} ({records_with_missing/len(df)*100:.1f}%)")
        print(f"\n  各字段缺失统计:")
        for field_cn, count in field_missing_count.items():
            if count > 0:
                print(f"    - {field_cn}: {count} 条 ({count/len(df)*100:.1f}%)")
        
        # 开始清洗
        print(f"\n[清洗] 开始数据清洗...")
        print(f"  策略: 尝试补全缺失数据 -> 删除无法补全的记录\n")
        
        删除的索引 = []
        补全的记录数 = 0
        处理进度 = 0
        
        for idx, row in df.iterrows():
            处理进度 += 1
            app_id = row['AppID']
            
            # 检查哪些字段缺失
            missing_fields = []
            for field_cn, field_type in self.critical_fields.items():
                if self.is_field_missing(row[field_cn], field_type):
                    missing_fields.append(field_type)
            
            # 如果有缺失,尝试补全
            if missing_fields:
                print(f"[{处理进度}/{len(df)}] AppID {app_id} - 缺失字段: {len(missing_fields)} 个")
                print(f"    缺失: {[k for k, v in self.critical_fields.items() if v in missing_fields]}")
                
                try:
                    补充数据 = self.fetch_missing_data(app_id, missing_fields)
                    
                    # 特殊处理: 如果仅有格式化价格缺失且爬取失败,填充 "Free"
                    if (missing_fields == ['price'] or (len(missing_fields) == 1 and missing_fields[0] == 'price')):
                        if not 补充数据 or '格式化价格' not in 补充数据:
                            print(f"    [价格] 爬取失败,填充 'Free'")
                            df.at[idx, '格式化价格'] = 'Free'
                            补全的记录数 += 1
                            continue
                    
                    if 补充数据:
                        print(f"    [补全] 成功补全 {len(补充数据)} 个字段")
                        # 更新数据
                        for field_cn, value in 补充数据.items():
                            df.at[idx, field_cn] = value
                    
                    # 再次检查是否还有缺失
                    still_missing = []
                    for field_cn, field_type in self.critical_fields.items():
                        if field_type in missing_fields:
                            if self.is_field_missing(df.at[idx, field_cn], field_type):
                                still_missing.append(field_cn)
                    
                    # 特殊处理: 在线人数相关字段
                    player_fields = ['current_players', 'peak_24h', 'peak_alltime']
                    missing_player_fields = [f for f in still_missing if self.critical_fields.get(f) in player_fields]
                    
                    if missing_player_fields:
                        # 检查是否三个字段都缺失
                        player_field_types = [self.critical_fields[f] for f in missing_player_fields]
                        
                        if set(player_field_types) == set(player_fields):
                            # 三个字段都缺失,删除该记录
                            print(f"    [删除] 在线人数相关字段全部缺失,标记删除")
                            删除的索引.append(idx)
                        else:
                            # 部分缺失,尝试估算
                            print(f"    [估算] 尝试估算缺失的在线人数字段...")
                            估算数据 = self.estimate_player_fields(df.loc[idx], player_field_types)
                            
                            if 估算数据:
                                print(f"    [成功] 成功估算 {len(估算数据)} 个字段")
                                for field_cn, value in 估算数据.items():
                                    df.at[idx, field_cn] = value
                                    if field_cn in missing_player_fields:
                                        missing_player_fields.remove(field_cn)
                                        still_missing.remove(field_cn)
                            
                            # 更新 still_missing
                            still_missing = [f for f in still_missing if f not in missing_player_fields or self.is_field_missing(df.at[idx, f], self.critical_fields[f])]
                    
                    # 特殊处理: 如果只剩价格缺失,填充 "Free"
                    if still_missing == ['格式化价格']:
                        print(f"    [价格] 仅价格缺失,填充 'Free'")
                        df.at[idx, '格式化价格'] = 'Free'
                        still_missing = []
                    
                    if still_missing:
                        print(f"    [警告] 仍有 {len(still_missing)} 个字段无法补全,标记删除")
                        print(f"    未补全字段: {still_missing}")
                        删除的索引.append(idx)
                    else:
                        补全的记录数 += 1
                    
                    # 请求延迟
                    time.sleep(1.5 + random.uniform(0, 1))
                    
                except Exception as e:
                    print(f"    [错误] 处理出错: {e},标记删除")
                    删除的索引.append(idx)
            else:
                if 处理进度 % 100 == 0:
                    print(f"[{处理进度}/{len(df)}] 数据完整,跳过")
        
        # 删除无法补全的记录
        print(f"\n[结果] 清洗结果:")
        print(f"  - 成功补全: {补全的记录数} 条")
        print(f"  - 无法补全(删除): {len(删除的索引)} 条")
        
        if 删除的索引:
            df_cleaned = df.drop(删除的索引).reset_index(drop=True)
        else:
            df_cleaned = df
        
        print(f"  - 最终保留: {len(df_cleaned)} 条 (原始: {len(df)} 条)")
        print(f"  - 删除比例: {len(删除的索引)/len(df)*100:.1f}%")
        
        # 保存清洗后的数据
        if output_file is None:
            base_name = input_file.rsplit('.', 1)[0]
            output_file = f"{base_name}_cleaned.csv"
        
        try:
            df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n[保存] 清洗后的数据已保存到: {output_file}")
            
            # 再次统计数据质量
            print(f"\n[质量] 清洗后数据质量:")
            for field_cn, field_type in self.critical_fields.items():
                missing_count = df_cleaned[field_cn].apply(
                    lambda x: self.is_field_missing(x, field_type)
                ).sum()
                print(f"  - {field_cn}: {missing_count} 条缺失 ({missing_count/len(df_cleaned)*100:.1f}%)")
            
        except Exception as e:
            print(f"[错误] 保存文件失败: {e}")


def main():
    """主函数"""
    import os
    
    cleaner = SteamDataCleaner()
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入输出文件路径(使用绝对路径)
    input_file = os.path.join(script_dir, "Source data.csv")
    output_file = os.path.join(script_dir, "Source data_cleaned.csv")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"[错误] 找不到输入文件")
        print(f"  期望路径: {input_file}")
        print(f"\n请确保 'Source data.csv' 文件位于以下目录:")
        print(f"  {script_dir}")
        return
    
    # 执行清洗
    cleaner.clean_data(input_file, output_file)
    
    print("\n" + "=" * 80)
    print("数据清洗完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
