from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache, wraps
import logging
import calendar
import json
import os
from werkzeug.utils import secure_filename
import shutil
import uuid
import threading
from time import sleep
import atexit
import numpy as np
from secrets import token_hex

app = Flask(__name__)

# 在 app 配置后添加
app.config.update(
    SESSION_COOKIE_SECURE=True,  # 只在 HTTPS 下发送 cookie
    SESSION_COOKIE_HTTPONLY=True,  # 防止 JavaScript 访问 cookie
    SESSION_COOKIE_SAMESITE='Lax',  # 防止 CSRF 攻击
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30)  # 设置会话過期时间
)

# 从环境变量获取密钥，如果没有则生成一個新的
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or token_hex(32)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # 添加文件處理器
        logging.StreamHandler()  # 保留控制台输出
    ]
)
logger = logging.getLogger(__name__)

# 添加文件上傳配置
UPLOAD_FOLDER = './datas/alipay_analysis'  # 临时文件根目录
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_dir():
    """获取当前会话的临时目录"""
    if 'user_id' not in session:
        # 使用更安全的方式生成用户ID
        session['user_id'] = f"user_{token_hex(16)}"
        # 记录会话创建时间
        session['created_at'] = datetime.now().timestamp()
        
    # 检查会话是否過期
    if 'created_at' in session:
        session_age = datetime.now().timestamp() - session['created_at']
        if session_age > 1800:  # 30分钟過期
            # 清理旧文件
            old_session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'])
            if os.path.exists(old_session_dir):
                shutil.rmtree(old_session_dir)
            # 重新生成会话
            session.clear()
            return get_session_dir()
    
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'])
    
    if not os.path.exists(session_dir):
        os.makedirs(session_dir, mode=0o700)  # 确保目录权限正确
    
    return session_dir

def user_cache(f):
    """用户级别的缓存装饰器"""
    cache = {}
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 获取当前用户ID
        user_id = session.get('user_id')
        if not user_id:
            return f(*args, **kwargs)
        
        # 检查缓存是否過期
        if user_id in cache:
            cache_time, cached_data = cache[user_id]
            if datetime.now().timestamp() - cache_time < 300:  # 5分钟缓存
                return cached_data
        
        # 执行函数并缓存结果
        result = f(*args, **kwargs)
        cache[user_id] = (datetime.now().timestamp(), result)
        
        # 清理其他用户的缓存
        current_time = datetime.now().timestamp()
        expired_keys = [k for k, v in cache.items() 
                       if current_time - v[0] > 300]
        for k in expired_keys:
            del cache[k]
        
        return result
    
    return decorated_function

@user_cache
def load_alipay_data():
    try:
        session_dir = get_session_dir()
        all_data = []
        
        # 读取会话目录中的所有CSV文件
        for filename in os.listdir(session_dir):
            if filename.endswith('.csv'):
                try:
                    filepath = os.path.join(session_dir, filename)
                    
                    # 读取CSV文件
                    df = pd.read_csv(filepath, encoding='utf-8')
                    
                    # 数据預處理
                    df['日期時間'] = pd.to_datetime(df['日期'] + ' ' + df['時間'])
                    df['月份'] = df['日期時間'].dt.strftime('%Y-%m')
                    df['日期'] = df['日期時間'].dt.strftime('%Y-%m-%d')
                    
                    # 處理金額
                    df['金額'] = df['金額'].astype(float)
                    
                    all_data.append(df)
                        
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {str(e)}")
                    continue
        
        if not all_data:
            raise FileNotFoundError("未找到任何帳單文件")
        
        # 合併所有數據
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('日期時間')
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    # try:
    #     session_dir = get_session_dir()
    #     all_data = []
        
    #     # 读取会话目录中的所有CSV文件
    #     for filename in os.listdir(session_dir):
    #         if filename.endswith('.csv'):
    #             try:
    #                 filepath = os.path.join(session_dir, filename)
                    
    #                 # 读取CSV文件
    #                 with open(filepath, encoding='gbk') as f:
    #                     lines = f.readlines()
    #                     header_row = None
    #                     status_row = None
    #                     for i, line in enumerate(lines):
    #                         if '描述' in line:
    #                             status_row = i
    #                         if '時間' in line:
    #                             header_row = i
    #                             break
                    
    #                 if header_row is not None:
    #                     df = pd.read_csv(filepath, encoding='gbk', skiprows=header_row)
                        
    #                     # 获取描述列
    #                     status_df = pd.read_csv(filepath, encoding='gbk', skiprows=status_row, nrows=1)
    #                     status_column = status_df.columns[0]
                        
    #                     # 数据预處理
    #                     df['時間'] = pd.to_datetime(df['時間'])
    #                     df['月份'] = df['時間'].dt.strftime('%Y-%m')
    #                     df['日期'] = df['時間'].dt.strftime('%Y-%m-%d')
                        
    #                     # 标记描述
    #                     df['記錄類型'] = df[status_column].isin(['退款成功', '交易关闭'])
    #                     df.loc[df['記錄類型'], '金額'] = -df.loc[df['記錄類型'], '金額']
                        
    #                     all_data.append(df)
                        
    #             except Exception as e:
    #                 logger.error(f"Error processing file {filename}: {str(e)}")
    #                 continue
        
    #     if not all_data:
    #         raise FileNotFoundError("未找到任何支付宝账单文件")
        
    #     # 合并所有数据
    #     combined_df = pd.concat(all_data, ignore_index=True)
    #     combined_df = combined_df.sort_values('時間')
        
    #     return combined_df
        
    # except Exception as e:
    #     logger.error(f"Error loading data: {str(e)}")
    #     raise

# def validate_dataframe(df):
#     required_columns = ['時間', '記錄類型', '金額', '主類別', '商品说明']
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         raise ValueError(f"数据缺少必需列: {', '.join(missing_columns)}")
    
#     # 验证数据类型
#     if not pd.api.types.is_numeric_dtype(df['金額']):
#         raise ValueError("'金額'列必须是数值类型")

def validate_dataframe(df):
    required_columns = ['帳戶', '幣種', '記錄類型', '主類別', '子類別', '金額', '手續費', '折扣', '名稱', '商家', '日期', '時間', '專案', '描述', '標籤', '對象']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"數據缺少必需列: {', '.join(missing_columns)}")
    
    # 驗證數據類型
    if not pd.api.types.is_numeric_dtype(df['金額']):
        raise ValueError("'金額'列必須是數值類型")

def check_data_exists(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 如果是 settings 页面，不需要检查数据
        if request.endpoint == 'settings':
            return f(*args, **kwargs)
            
        if 'user_id' not in session:
            return redirect(url_for('settings'))
            
        session_dir = get_session_dir()
        has_data = False
        print("=== session_dir ===", session_dir)
        if os.path.exists(session_dir):
            for filename in os.listdir(session_dir):
                if filename.endswith('.csv'):
                    has_data = True
                    break
        print("=== has_data ===", has_data)
        if not has_data:
            return redirect(url_for('settings'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', active_page='index')

@app.route('/yearly')
@check_data_exists
def yearly():
    return render_template('yearly.html', active_page='yearly')

@app.route('/monthly')
@check_data_exists
def monthly():
    return render_template('monthly.html', active_page='monthly')

@app.route('/category')
@check_data_exists
def category():
    return render_template('category.html', active_page='category')

@app.route('/time')
@check_data_exists
def time():
    return render_template('time.html', active_page='time')

@app.route('/transactions')
@check_data_exists
def transactions():
    return render_template('transactions.html', active_page='transactions')

@app.route('/insights')
@check_data_exists
def insights():
    return render_template('insights.html', active_page='insights')

@app.route('/analysis')
@check_data_exists
def analysis():
    print("=== HiHi ===")
    return render_template('analysis.html', active_page='analysis')

@app.route('/api/analysis')
def get_analysis():
    try:
        df = load_alipay_data()
        
        # 获取年份参数
        year = request.args.get('year', type=int)
        if year:
            df = df[df['時間'].dt.year == year]
        
        # 商家分析
        merchant_analysis = analyze_merchants(df)
        
        # 消费场景分析 
        scenario_analysis = analyze_scenarios(df)
        
        # 消费习惯分析
        habit_analysis = analyze_habits(df)
        
        # 智能标签
        tags = generate_smart_tags(df)
        
        # 分析支付方式
        payment_analysis = analyze_payment_methods(df)
        
        return jsonify({
            'success': True,
            'data': {
                'merchant_analysis': merchant_analysis,
                'scenario_analysis': scenario_analysis,
                'habit_analysis': habit_analysis,
                'tags': tags,
                'payment_analysis': payment_analysis
            }
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/monthly_analysis')
def monthly_analysis():
    try:
        df = load_alipay_data()
        year, month = request.args.get('year', type=int), request.args.get('month', type=int)
        min_amount = request.args.get('min_amount', type=float)
        max_amount = request.args.get('max_amount', type=float)
        
        # 获取当前月份数据
        current_month_df = df[
            (df['時間'].dt.year == year) & 
            (df['時間'].dt.month == month)
        ]
        
        # 获取上月数据
        last_month = month - 1 if month > 1 else 12
        last_year = year if month > 1 else year - 1
        last_month_df = df[
            (df['時間'].dt.year == last_year) & 
            (df['時間'].dt.month == last_month)
        ]
        
        # 应用金額筛选
        if min_amount:
            current_month_df = current_month_df[current_month_df['金額'] >= min_amount]
            last_month_df = last_month_df[last_month_df['金額'] >= min_amount]
        if max_amount:
            current_month_df = current_month_df[current_month_df['金額'] < max_amount]
            last_month_df = last_month_df[last_month_df['金額'] < max_amount]
        
        # 處理收入和支出数据
        current_expense_df = current_month_df[
            (current_month_df['記錄類型'] == '支出') & 
            (current_month_df['記錄類型'] != '退款')
        ]
        current_income_df = current_month_df[
            (current_month_df['記錄類型'] == '收入') & 
            (current_month_df['記錄類型'] != '退款')
        ]
        
        # 计算统计数据
        current_expense = current_expense_df['金額'].sum()
        current_income = current_income_df['金額'].sum()
        current_balance = current_income - current_expense
        
        # 计算上月数据
        last_expense = last_month_df[
            (last_month_df['記錄類型'] == '支出') & 
            (last_month_df['記錄類型'] != '退款')
        ]['金額'].sum()
        last_income = last_month_df[
            (last_month_df['記錄類型'] == '收入') & 
            (last_month_df['記錄類型'] != '退款')
        ]['金額'].sum()
        last_balance = last_income - last_expense
        
        # 按日期统计
        daily_expenses = current_expense_df.groupby(
            current_expense_df['時間'].dt.date
        )['金額'].sum()
        daily_incomes = current_income_df.groupby(
            current_income_df['時間'].dt.date
        )['金額'].sum()
        
        # 计算分类统计
        expense_categories = current_expense_df.groupby('主類別')['金額'].sum()
        income_categories = current_income_df.groupby('主類別')['金額'].sum()
        
        # 生成当月所有日期
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        all_dates = [
            datetime(year, month, day).date() 
            for day in range(1, last_day + 1)
        ]
        
        # 补充所有日期，缺失的填充0
        daily_expenses = daily_expenses.reindex(all_dates, fill_value=0)
        daily_incomes = daily_incomes.reindex(all_dates, fill_value=0)
        
        return jsonify({
            'success': True,
            'data': {
                'stats': {
                    'balance': float(current_balance),
                    'total_expense': float(current_expense),
                    'total_income': float(current_income),
                    'expense_count': int(len(current_expense_df)),
                    'income_count': int(len(current_income_df)),
                    'comparisons': {
                        'balance': {
                            'change': float(current_balance - last_balance),
                            'rate': float((current_balance - last_balance) / abs(last_balance) * 100) if last_balance != 0 else None
                        },
                        'expense': {
                            'change': float(current_expense - last_expense),
                            'rate': float((current_expense - last_expense) / last_expense * 100) if last_expense != 0 else None
                        },
                        'income': {
                            'change': float(current_income - last_income),
                            'rate': float((current_income - last_income) / last_income * 100) if last_income != 0 else None
                        }
                    }
                },
                'daily_data': {
                    'expense': {
                        'dates': [d.strftime('%Y-%m-%d') for d in all_dates],
                        'amounts': daily_expenses.values.tolist()
                    },
                    'income': {
                        'dates': [d.strftime('%Y-%m-%d') for d in all_dates],
                        'amounts': daily_incomes.values.tolist()
                    }
                },
                'categories': {
                    'expense': {
                        'names': expense_categories.index.tolist(),
                        'amounts': expense_categories.values.tolist()
                    },
                    'income': {
                        'names': income_categories.index.tolist(),
                        'amounts': income_categories.values.tolist()
                    }
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in monthly analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/category_expenses')
def category_expenses():
    df = load_alipay_data()
    
    # 计算分类支出
    category_stats = df[df['記錄類型'] == '支出'].groupby('主類別').agg({
        '金額': 'sum'
    }).sort_values('金額', ascending=False)
    
    # 准备ECharts数据格式
    data = {
        'categories': category_stats.index.tolist(),
        'amounts': category_stats['金額'].tolist()
    }
    
    return jsonify(data)

@app.route('/api/transactions')
def get_transactions():
    try:
        df = load_alipay_data()
        
        # 獲取分頁參數
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)  # 默認每頁20條
        
        # 獲取篩選參數
        year = request.args.get('year', type=int)
        month = request.args.get('month', type=int)
        date = request.args.get('date')
        hour = request.args.get('hour', type=int)
        category = request.args.get('category')
        min_amount = request.args.get('min_amount', type=float)
        max_amount = request.args.get('max_amount', type=float)
        
        # 獲取交易類型參數（收入/支出）
        type = request.args.get('type')
        
        # 應用篩選條件
        if type:
            df = df[df['記錄類型'] == type]  # 根據收入/支出類型篩選
        
        if year:
            df = df[df['日期時間'].dt.year == year]
        if month:
            df = df[df['日期時間'].dt.month == month]
        if date:
            df = df[df['日期'] == date]
        if hour is not None:
            df = df[df['日期時間'].dt.hour == hour]
        if category:
            df = df[df['主類別'] == category]
        if min_amount:
            df = df[df['金額'] >= min_amount]
        if max_amount:
            df = df[df['金額'] <= max_amount]
            
        # 排除"不計收支"的交易
        df = df[df['記錄類型'].isin(['收入', '支出'])]
        
        # 按時間倒序排序
        df = df.sort_values('日期時間', ascending=False)
        
        # 計算總記錄數和總頁數
        total_records = len(df)
        total_pages = (total_records + per_page - 1) // per_page
        
        # 確保頁碼在有效範圍內
        page = max(1, min(page, total_pages))
        
        # 計算當前頁的數據範圍
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_records)
        
        # 獲取當前頁的數據
        page_df = df.iloc[start_idx:end_idx]
        
        # 轉換為列表，處理 NaN 值
        transactions = []
        for _, row in page_df.iterrows():
            transactions.append({
                'time': row['日期時間'].strftime('%Y-%m-%d %H:%M:%S'),
                'description': str(row['名稱'] if pd.notna(row['名稱']) else row['子類別']),
                'category': str(row['主類別']),
                'type': str(row['記錄類型']),
                'amount': float(row['金額']),
                'status': str(row['專案'] if pd.notna(row['專案']) else '無專案'),
                'counterparty': str(row.get('對象', '')) if pd.notna(row.get('對象')) else ''
            })
        
        # 返回數據，包含分頁信息
        return jsonify({
            'success': True,
            'transactions': transactions,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_records': total_records
            }
        })
        
    except Exception as e:
        logger.error(f"獲取交易記錄時出錯: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'獲取交易記錄失敗: {str(e)}'
        }), 500

# @app.route('/api/transactions')
# def get_transactions():
#     try:
#         df = load_alipay_data()
        
#         # 获取分页参数
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 20, type=int)  # 默认每页20条
        
#         # 获取筛选参数
#         year = request.args.get('year', type=int)
#         month = request.args.get('month', type=int)
#         date = request.args.get('date')
#         hour = request.args.get('hour', type=int)
#         category = request.args.get('category')
#         min_amount = request.args.get('min_amount', type=float)
#         max_amount = request.args.get('max_amount', type=float)
        
#         # 获取交易类型参数（收入/支出）
#         type = request.args.get('type')
        
#         # 应用筛选条件
#         if type:
#             df = df[df['記錄類型'] == type]  # 根据收入/支出类型筛选
        
#         if year:
#             df = df[df['時間'].dt.year == year]
#         if month:
#             df = df[df['時間'].dt.month == month]
#         if date:
#             df = df[df['日期'] == date]
#         if hour is not None:
#             df = df[df['時間'].dt.hour == hour]
#         if category:
#             df = df[df['主類別'] == category]
#         if min_amount:
#             df = df[df['金額'] >= min_amount]
#         if max_amount:
#             df = df[df['金額'] <= max_amount]
            
#         # 排除"不计收支"的交易
#         df = df[df['記錄類型'].isin(['收入', '支出'])]
        
#         # 排除退款交易
#         df = df[~df['記錄類型']]
        
#         # 按时间倒序排序
#         df = df.sort_values('時間', ascending=False)
        
#         # 计算总记录数和总页数
#         total_records = len(df)
#         total_pages = (total_records + per_page - 1) // per_page
        
#         # 确保页码在有效范围内
#         page = max(1, min(page, total_pages))
        
#         # 计算当前页的数据范围
#         start_idx = (page - 1) * per_page
#         end_idx = min(start_idx + per_page, total_records)
        
#         # 获取当前页的数据
#         page_df = df.iloc[start_idx:end_idx]
        
#         # 转换为列表，處理 NaN 值
#         transactions = []
#         for _, row in page_df.iterrows():
#             transactions.append({
#                 'time': row['時間'].strftime('%Y-%m-%d %H:%M:%S'),
#                 'description': str(row['商品说明']),
#                 'category': str(row['主類別']),
#                 'type': str(row['記錄類型']),
#                 'amount': float(row['金額']),
#                 'status': str(row['描述']),
#                 'counterparty': str(row.get('商家', '')) if pd.notna(row.get('商家')) else ''
#             })
        
#         # 返回数据，包含分页信息
#         return jsonify({
#             'success': True,
#             'transactions': transactions,
#             'pagination': {
#                 'current_page': page,
#                 'per_page': per_page,
#                 'total_pages': total_pages,
#                 'total_records': total_records
#             }
#         })
        
#     except Exception as e:
#         logger.error(f"获取交易记录时出错: {str(e)}", exc_info=True)
#         return jsonify({
#             'success': False,
#             'error': f'获取交易记录失败: {str(e)}'
#         }), 500

@app.route('/api/summary')
def summary():
    df = load_alipay_data()
    
    # 获取当前自然月
    current_date = datetime.now()
    current_month = current_date.strftime('%Y-%m')
    
    # 计算基础统计信息
    expense_df = df[df['記錄類型'] == '支出']
    total_expense = expense_df['金額'].sum()
    total_income = df[df['記錄類型'] == '收入']['金額'].sum()
    
    # 按月份分组计算支出
    monthly_expenses = expense_df.groupby('月份')['金額'].sum()
    
    # 获取最新月份的支出
    latest_month = monthly_expenses.index[-1]
    latest_month_expense = monthly_expenses[latest_month]
    
    # 获取当前自然月的支出（如果有数据的话）
    current_month_expense = monthly_expenses.get(current_month)
    
    # 确定要显示的月份和支出金額
    display_month = current_month if current_month_expense is not None else latest_month
    display_expense = current_month_expense if current_month_expense is not None else latest_month_expense
    
    # 计算环比变化（与上一個月相比）
    if len(monthly_expenses) > 1:
        prev_month_expense = monthly_expenses.iloc[-2]
    else:
        prev_month_expense = display_expense
    
    return jsonify({
        'total_expense': round(total_expense, 2),
        'total_income': round(total_income, 2),
        'balance': round(total_income - total_expense, 2),
        'monthly_avg': round(monthly_expenses.mean(), 2),
        'current_month_expense': round(display_expense, 2),
        'prev_monthly_avg': round(prev_month_expense, 2),
        'month_count': len(monthly_expenses),
        'transaction_count': len(expense_df),
        'current_month': display_month,
        'has_current_month_data': current_month_expense is not None
    })

@app.route('/api/daily_data')
def daily_data():
    """获取热力图数据，支持年份筛选"""
    try:
        df = load_alipay_data()
        df['時間'] = pd.to_datetime(df['時間'], errors='coerce')

        year = request.args.get('year', type=int)
        filter_type = request.args.get('filter', 'all')

        if filter_type == 'large':
            df = df[df['金額'] > 1000]
        elif filter_type == 'small':
            df = df[df['金額'] <= 1000]
            
        if year:
            df = df[df['時間'].dt.year == year]

        df = df[df['記錄類型'].isin(['收入', '支出'])]

        daily_data = df.groupby(['日期', '記錄類型']).agg({'金額': 'sum', '時間': 'count'}).reset_index()

        expense_data, income_data, transaction_data = [], [], []

        for date, group in daily_data.groupby('日期'):
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')  # 修正日期格式
            
            expense = group[group['記錄類型'] == '支出']
            if not expense.empty:
                expense_value = abs(float(expense['金額'].iloc[0]))  
                expense_data.append([date_str, expense_value])
                
            income = group[group['記錄類型'] == '收入']
            if not income.empty:
                income_value = max(float(income['金額'].iloc[0]), 0)  
                income_data.append([date_str, income_value])
                
            transaction_count = group['時間'].sum()
            transaction_data.append([date_str, int(transaction_count)])

        expense_amounts = [abs(x[1]) for x in expense_data]
        income_amounts = [max(x[1], 0) for x in income_data]

        expense_quantiles, income_quantiles = [], []
        if expense_amounts:
            expense_quantiles = [round(float(x), 2) for x in np.quantile(expense_amounts, [0.2, 0.4, 0.6, 0.8])]
        if income_amounts:
            income_quantiles = [round(float(x), 2) for x in np.quantile(income_amounts, [0.2, 0.4, 0.6, 0.8])]

        return jsonify({
            'expense': expense_data,
            'income': income_data,
            'transaction': transaction_data,
            'expense_quantiles': expense_quantiles,
            'income_quantiles': income_quantiles
        })

    except Exception as e:
        logger.error(f"Error in daily data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/category_detail/<month>/<category>')
def category_detail(month, category):
    df = load_alipay_data()
    
    # 获取指定月份和分类的支出明细，添加商家列
    details = df[
        (df['月份'] == month) & 
        (df['主類別'] == category) & 
        (df['記錄類型'] == '支出')
    ].sort_values('金額', ascending=False)[
        ['時間', '商品说明', '商家', '金額', '描述']
    ].to_dict('records')
    
    # 格式化数据
    formatted_details = [{
        'time': detail['時間'].strftime('%Y-%m-%d %H:%M:%S'),
        'description': detail['商品说明'],
        'counterparty': detail['商家'],  # 添加商家
        'amount': round(float(detail['金額']), 2),
        'status': detail['描述']
    } for detail in details]
    
    return jsonify(formatted_details)

@app.route('/api/top_transactions')
def get_top_transactions():
    """获取大额交易记录"""
    try:
        # 获取查询参数
        limit = int(request.args.get('limit', 10))  # 默认返回前10条
        min_amount = float(request.args.get('min_amount', 1000))  # 默认1000元以上
        
        # 加载数据
        df = load_alipay_data()
        
        # 筛选支出交易
        expense_df = df[df['記錄類型'] == '支出'].copy()
        
        # 筛选大额交易
        large_transactions = expense_df[expense_df['金額'] >= min_amount]
        
        # 按金額排序并限制数量
        top_transactions = large_transactions.nlargest(limit, '金額')
        
        # 转换为列表
        transactions = []
        for _, row in top_transactions.iterrows():
            transactions.append({
                'time': row['時間'].strftime('%Y-%m-%d %H:%M:%S'),
                'date': row['時間'].strftime('%Y-%m-%d'),
                'category': row['主類別'],
                'description': row['商品说明'],
                'amount': float(row['金額']),
                'status': row['描述'],
                'counterparty': row.get('商家', '')
            })
        
        return jsonify({
            'success': True,
            'transactions': transactions
        })
        
    except Exception as e:
        logger.error(f"获取大额交易记录时出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'获取大额交易记录失败: {str(e)}'
        }), 500

@app.route('/api/category_trend/<category>')
def category_trend(category):
    df = load_alipay_data()
    
    # 获取指定分类的月度数据
    category_df = df[
        (df['記錄類型'] == '支出') & 
        (df['主類別'] == category)
    ]
    
    # 按月份分组计算各项指标
    monthly_stats = category_df.groupby('月份').agg({
        '金額': ['sum', 'count', 'mean'],  # 总金額、交易次数、平均金額
        '時間': lambda x: len(x.dt.date.unique())  # 有交易的天数
    }).round(2)
    
    # 重命名列
    monthly_stats.columns = ['total', 'transactions', 'avg_amount', 'active_days']
    
    # 计算日均支出（总金額/当月有交易的天数）
    monthly_stats['daily_avg'] = (monthly_stats['total'] / monthly_stats['active_days']).round(2)
    
    # 计算环比变化率
    monthly_stats['mom_rate'] = (monthly_stats['total'].pct_change() * 100).round(2)
    
    # 计算占比
    total_expense = df[
        (df['記錄類型'] == '支出')
    ].groupby('月份')['金額'].sum()
    monthly_stats['percentage'] = (monthly_stats['total'] / total_expense * 100).round(2)
    
    # 准备返回数据
    result = {
        'months': monthly_stats.index.tolist(),
        'total': monthly_stats['total'].tolist(),
        'transactions': monthly_stats['transactions'].tolist(),
        'avg_amount': monthly_stats['avg_amount'].tolist(),
        'daily_avg': monthly_stats['daily_avg'].tolist(),
        'mom_rate': monthly_stats['mom_rate'].fillna(0).tolist(),
        'percentage': monthly_stats['percentage'].tolist(),
        'summary': {
            'total_amount': category_df['金額'].sum().round(2),
            'total_transactions': len(category_df),
            'max_month': monthly_stats['total'].idxmax(),
            'max_amount': monthly_stats['total'].max().round(2),
            'min_month': monthly_stats['total'].idxmin(),
            'min_amount': monthly_stats['total'].min().round(2),
            'avg_monthly': monthly_stats['total'].mean().round(2)
        }
    }
    
    return jsonify(result)

# @app.route('/api/time_analysis')
# def time_analysis():
    """获取时间分析数据，支持年份筛选"""
    try:
        df = load_alipay_data()
        
        # 获取查询参数
        year = request.args.get('year', type=int)
        filter_type = request.args.get('filter', 'all')
        
        # 只分析支出数据
        expense_df = df[df['記錄類型'] == '支出']
        
        # 如果指定了年份，過滤对应年份的数据
        if year:
            expense_df = expense_df[expense_df['時間'].dt.year == year]
            
        # 根据筛选条件過滤数据 - 移到这里，在计算统计数据之前過滤
        if filter_type == 'large':
            expense_df = expense_df[expense_df['金額'] > 1000]
        elif filter_type == 'small':
            expense_df = expense_df[expense_df['金額'] <= 1000]
        
        # 1. 计算日内时段分布
        expense_df['hour'] = expense_df['時間'].dt.hour
        hourly_stats = expense_df.groupby('hour').agg({
            '金額': 'sum',
            '時間': 'count'
        }).reset_index()
        
        # 确保所有小时都有数据，没有数据的填充0
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_stats = pd.merge(all_hours, hourly_stats, on='hour', how='left').fillna(0)
        
        hourly_data = {
            'amounts': hourly_stats['金額'].round(2).tolist(),
            'counts': hourly_stats['時間'].tolist()
        }
        
        # 2. 计算工作日/周末分布
        expense_df['is_weekend'] = expense_df['時間'].dt.dayofweek.isin([5, 6])
        category_weekday = {}
        
        for category in expense_df['主類別'].unique():
            category_df = expense_df[expense_df['主類別'] == category]
            
            if len(category_df) == 0:
                continue
                
            weekday_amount = category_df[~category_df['is_weekend']]['金額'].sum()
            weekend_amount = category_df[category_df['is_weekend']]['金額'].sum()
            total_amount = weekday_amount + weekend_amount
            
            if total_amount == 0:
                continue
                
            weekday_count = len(category_df[~category_df['is_weekend']])
            weekend_count = len(category_df[category_df['is_weekend']])
            
            category_weekday[category] = {
                'weekday': {
                    'amount': float(weekday_amount),
                    'count': int(weekday_count),
                    'percentage': round(weekday_amount / total_amount * 100, 1)
                },
                'weekend': {
                    'amount': float(weekend_amount),
                    'count': int(weekend_count),
                    'percentage': round(weekend_amount / total_amount * 100, 1)
                }
            }
        
        # 按总金額排序
        sorted_categories = sorted(
            category_weekday.items(),
            key=lambda x: x[1]['weekday']['amount'] + x[1]['weekend']['amount'],
            reverse=True
        )
        category_weekday = dict(sorted_categories)
        
        return jsonify({
            'hourly': hourly_data,
            'weekday_weekend': category_weekday
        })
        
    except Exception as e:
        logger.error(f"Error in time analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/time_analysis')
def time_analysis():
    """获取时间分析数据，支持年份筛选"""
    try:
        df = load_alipay_data()

        # 确保 "時間" 是 datetime 类型
        df['時間'] = pd.to_datetime(df['時間'], errors='coerce')

        # 获取查询参数
        year = request.args.get('year', type=int)
        filter_type = request.args.get('filter', 'all')

        # 只分析支出数据，确保创建副本，避免 SettingWithCopyWarning
        expense_df = df[df['記錄類型'] == '支出'].copy()

        # 确保 "金額" 为正数
        expense_df['金額'] = expense_df['金額'].abs()

        # 如果指定了年份，過滤对应年份的数据
        if year:
            expense_df = expense_df[expense_df['時間'].dt.year == year]

        # 根据筛选条件過滤数据
        if filter_type == 'large':
            expense_df = expense_df[expense_df['金額'] > 1000]
        elif filter_type == 'small':
            expense_df = expense_df[expense_df['金額'] <= 1000]

        # 1. 计算日内时段分布
        expense_df['hour'] = expense_df['時間'].dt.hour
        hourly_stats = expense_df.groupby('hour').agg({
            '金額': 'sum',
            '時間': 'count'
        }).reset_index()

        # 确保所有小时都有数据
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_stats = pd.merge(all_hours, hourly_stats, on='hour', how='left').fillna(0)

        hourly_data = {
            'amounts': hourly_stats['金額'].round(2).tolist(),
            'counts': hourly_stats['時間'].astype(int).tolist()
        }

        # 2. 计算工作日/周末分布
        expense_df['is_weekend'] = expense_df['日期'].dt.dayofweek.isin([5, 6])
        category_weekday = {}

        for category in expense_df['主類別'].unique():
            category_df = expense_df[expense_df['主類別'] == category]

            if category_df.empty:
                continue

            weekday_amount = category_df[~category_df['is_weekend']]['金額'].sum()
            weekend_amount = category_df[category_df['is_weekend']]['金額'].sum()
            total_amount = weekday_amount + weekend_amount

            if total_amount == 0:
                continue  # 避免分母为 0

            weekday_count = len(category_df[~category_df['is_weekend']])
            weekend_count = len(category_df[category_df['is_weekend']])

            category_weekday[category] = {
                'weekday': {
                    'amount': float(weekday_amount),
                    'count': int(weekday_count),
                    'percentage': round(weekday_amount / max(total_amount, 1e-9) * 100, 1)  # 避免 ZeroDivisionError
                },
                'weekend': {
                    'amount': float(weekend_amount),
                    'count': int(weekend_count),
                    'percentage': round(weekend_amount / max(total_amount, 1e-9) * 100, 1)  # 避免 ZeroDivisionError
                }
            }

        # 按总金额排序
        sorted_categories = sorted(
            category_weekday.items(),
            key=lambda x: x[1]['weekday']['amount'] + x[1]['weekend']['amount'],
            reverse=True
        )
        category_weekday = dict(sorted_categories)

        return jsonify({
            'hourly': hourly_data,
            'weekday_weekend': category_weekday
        })

    except Exception as e:
        logger.error(f"Error in time analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/filtered_monthly_analysis')
def filtered_monthly_analysis():
    df = load_alipay_data()
    filter_type = request.args.get('filter', 'all')
    
    # 在原始交易数据层面进行過滤
    expense_df = df[df['記錄類型'] == '支出']
    if filter_type == 'large':
        expense_df = expense_df[expense_df['金額'] > 1000]
    elif filter_type == 'small':
        expense_df = expense_df[expense_df['金額'] <= 1000]
    
    # 使用過滤后的数据计算月度统计
    monthly_stats = expense_df.groupby('月份').agg({
        '金額': ['sum', 'count', 'mean'],  # 总金額、交易次数、平均金額
        '時間': lambda x: len(x.dt.date.unique())  # 有交易的天数
    }).round(2)
    
    monthly_stats.columns = ['total', 'count', 'avg_amount', 'active_days']
    
    # 计算日均支出
    monthly_stats['daily_avg'] = (monthly_stats['total'] / monthly_stats['active_days']).round(2)
    
    # 计算环比变化率
    monthly_stats['mom_rate'] = (monthly_stats['total'].pct_change() * 100).round(2)
    
    # 计算3個月移动平均
    monthly_stats['moving_avg'] = monthly_stats['total'].rolling(3, min_periods=1).mean().round(2)
    
    # 计算分类支出
    category_expenses = expense_df.pivot_table(
        index='月份',
        columns='主類別',
        values='金額',
        aggfunc='sum',
        fill_value=0
    )
    
    return jsonify({
        'months': monthly_stats.index.tolist(),
        'total_expenses': monthly_stats['total'].tolist(),
        'transaction_counts': monthly_stats['count'].tolist(),
        'daily_averages': monthly_stats['daily_avg'].tolist(),
        'mom_rates': monthly_stats['mom_rate'].fillna(0).tolist(),
        'moving_averages': monthly_stats['moving_avg'].tolist(),
        'categories': category_expenses.columns.tolist(),
        'category_expenses': category_expenses.values.tolist()
    })

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(404)
def not_found_error(error):
    return "页面未找到 - 404", 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {str(error)}")
    return jsonify({
        'success': False,
        'error': '服务器内部错误，请稍后重试'
    }), 500

@app.route('/api/overview_data')
def get_overview_data():
    try:
        df = load_alipay_data()
        
        # 获取筛选类型
        filter_type = request.args.get('filter', 'all')
        year = request.args.get('year', None)
        
        if year is None:
            year = str(df['時間'].max().year)
        
        # 筛选指定年份的数据
        year_df = df[df['時間'].dt.year == int(year)]
        
        # 根据金額筛选数据
        if filter_type == 'large':
            year_df = year_df[year_df['金額'].abs() >= 1000]
        elif filter_type == 'small':
            year_df = year_df[year_df['金額'].abs() < 1000]
        
        # 获取所有可用的年份列表
        available_years = sorted(df['時間'].dt.year.unique().tolist(), reverse=True)
        
        # 计算年度统计数据
        expense_df = year_df[
            (year_df['記錄類型'] == '支出') & 
            (year_df['記錄類型'] != '退款')
        ]
        income_df = year_df[
            (year_df['記錄類型'] == '收入') & 
            (year_df['記錄類型'] != '退款')
        ]
        
        # 年度统计，所有金額保留2位小数
        yearly_stats = {
            'total_expense': round(expense_df['金額'].sum(), 2),
            'total_income': round(income_df['金額'].sum(), 2),
            'balance': round(income_df['金額'].sum() - expense_df['金額'].sum(), 2),
            'expense_count': len(expense_df),
            'income_count': len(income_df),
            'total_count': len(expense_df) + len(income_df),
            'active_days': len(year_df['日期'].unique()),
            'avg_transaction': round(expense_df['金額'].mean(), 2) if len(expense_df) > 0 else 0,
            'avg_daily_expense': round(expense_df['金額'].sum() / 365, 2),
            'avg_monthly_income': round(income_df['金額'].sum() / 12, 2),
            'expense_ratio': round(expense_df['金額'].sum() / income_df['金額'].sum() * 100, 2) if income_df['金額'].sum() > 0 else 0
        }
        
        # 确保包含所有月份（1-12月）
        all_months = [f'{year}-{str(month).zfill(2)}' for month in range(1, 13)]
        
        # 按月统计支出
        monthly_data = expense_df.groupby('月份')['金額'].sum().round(2)
        
        # 创建包含所有月份的完整数据框
        monthly_stats = pd.DataFrame(index=all_months)
        monthly_stats['total'] = monthly_data
        
        # 填充缺失值
        monthly_stats = monthly_stats.fillna(0)
        
        # 计算分类统计
        category_stats = expense_df.groupby('主類別')['金額'].sum().round(2).sort_values(ascending=False)
        
        return jsonify({
            'available_years': available_years,
            'current_year': year,
            'yearly_stats': yearly_stats,
            'months': all_months,
            'amounts': monthly_stats['total'].tolist(),
            'categories': category_stats.index.tolist(),
            'amounts_by_category': category_stats.values.tolist()
        })
        
    except Exception as e:
        print(f"API错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/monthly_data')
def get_monthly_data():
    try:
        df = load_alipay_data()
        
        # 获取所有可用月份
        available_months = sorted(df['月份'].unique().tolist(), reverse=True)
        
        # 从请求参数获取年月，如果没有指定则使用最新的可用月份
        latest_month = available_months[0]
        default_year = int(latest_month.split('-')[0])
        default_month = int(latest_month.split('-')[1])
        
        current_year = request.args.get('year', default_year, type=int)
        current_month = request.args.get('month', default_month, type=int)
        
        logger.info(f"请求月度数据: {current_year}-{current_month}")
        
        # 获取当前月份数据
        current_month_str = f"{current_year}-{current_month:02d}"
        current_month_df = df[df['月份'] == current_month_str].copy()
        
        # 计算上個月的年份和月份
        if current_month == 1:
            last_month_year = current_year - 1
            last_month = 12
        else:
            last_month_year = current_year
            last_month = current_month - 1
            
        # 获取上個月数据
        last_month_str = f"{last_month_year}-{last_month:02d}"
        last_month_df = df[df['月份'] == last_month_str].copy()
        
        # 获取筛选参数
        filter_type = request.args.get('filter', 'all')
        
        # 根据筛选类型過滤数据
        if filter_type == 'large':
            current_month_df = current_month_df[current_month_df['金額'] >= 1000]  # 修改为1000元
            last_month_df = last_month_df[last_month_df['金額'] >= 1000] if not last_month_df.empty else last_month_df
        elif filter_type == 'small':
            current_month_df = current_month_df[current_month_df['金額'] < 1000]  # 修改为1000元
            last_month_df = last_month_df[last_month_df['金額'] < 1000] if not last_month_df.empty else last_month_df
        
        # 计算当前月统计
        current_stats = calculate_monthly_stats(current_month_df)
        monthly_stats = {
            **current_stats,  # 保持原有统计数据
        }
        
        # 只有当存在上月数据时才计算环比
        if not last_month_df.empty:
            last_month_stats = calculate_monthly_stats(last_month_df)
            
            # 添加环比变化数据
            balance_change_rate = calculate_change_rate(current_stats['balance'], last_month_stats['balance'])
            expense_change_rate = calculate_change_rate(current_stats['total_expense'], last_month_stats['total_expense'])
            income_change_rate = calculate_change_rate(current_stats['total_income'], last_month_stats['total_income'])
            transaction_change_rate = calculate_change_rate(current_stats['total_count'], last_month_stats['total_count'])
            
            monthly_stats.update({
                'balance_change': float(current_stats['balance'] - last_month_stats['balance']),
                'expense_change': float(current_stats['total_expense'] - last_month_stats['total_expense']),
                'income_change': float(current_stats['total_income'] - last_month_stats['total_income']),
                'transaction_change': int(current_stats['total_count'] - last_month_stats['total_count']),
                'balance_change_rate': float(balance_change_rate) if balance_change_rate is not None else None,
                'expense_change_rate': float(expense_change_rate) if expense_change_rate is not None else None,
                'income_change_rate': float(income_change_rate) if income_change_rate is not None else None,
                'transaction_change_rate': float(transaction_change_rate) if transaction_change_rate is not None else None
            })
        else:
            monthly_stats.update({
                'balance_change': None,
                'expense_change': None,
                'income_change': None,
                'transaction_change': None,
                'balance_change_rate': None,
                'expense_change_rate': None,
                'income_change_rate': None,
                'transaction_change_rate': None
            })
        
        # 获取当月的日期范围
        month_dates = pd.date_range(
            start=f"{current_year}-{current_month:02d}-01",
            end=pd.Timestamp(f"{current_year}-{current_month:02d}-01") + pd.offsets.MonthEnd(1),
            freq='D'
        )
        
        # 创建日期索引的数据框
        daily_stats = pd.DataFrame(index=month_dates.strftime('%Y-%m-%d'))
        
        # 计算每日支出和收入
        expense_df = current_month_df[current_month_df['記錄類型'] == '支出']
        income_df = current_month_df[current_month_df['記錄類型'] == '收入']
        
        daily_expense = expense_df.groupby('日期')['金額'].sum()
        daily_income = income_df.groupby('日期')['金額'].sum()
        
        daily_stats['expense'] = daily_expense
        daily_stats['income'] = daily_income
        daily_stats = daily_stats.fillna(0)
        
        # 计算分类统计
        category_stats = current_month_df[
            (current_month_df['記錄類型'] == '支出') & 
            (current_month_df['記錄類型'] != '退款')
        ].groupby('主類別')['金額'].agg([
            ('total', 'sum'),
            ('count', 'count')
        ]).round(2)
        category_stats = category_stats.sort_values('total', ascending=False)
        
        return jsonify({
            'available_months': available_months,
            'current_month': current_month_str,
            'monthly_stats': monthly_stats,
            'daily_data': {
                'dates': daily_stats.index.tolist(),
                'expenses': daily_stats['expense'].tolist(),
                'incomes': daily_stats['income'].tolist()
            },
            'categories': {
                'names': category_stats.index.tolist(),
                'amounts': category_stats['total'].tolist(),
                'counts': category_stats['count'].tolist()
            }
        })
        
    except Exception as e:
        logger.error(f"處理月度数据时出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def calculate_monthly_stats(df):
    """计算月度统计数据"""
    if df.empty:
        return {
            'balance': 0,
            'total_expense': 0,
            'total_income': 0,
            'expense_count': 0,
            'income_count': 0,
            'total_count': 0,
            'active_days': 0,
            'avg_transaction': 0,
            'avg_daily_expense': 0,
            'expense_ratio': 0
        }
    
    # 過滤有效交易
    expense_df = df[
        (df['記錄類型'] == '支出') & 
        (df['記錄類型'] != '退款')
    ]
    income_df = df[
        (df['記錄類型'] == '收入') & 
        (df['記錄類型'] != '退款')
    ]
    
    total_expense = expense_df['金額'].sum() if not expense_df.empty else 0
    total_income = income_df['金額'].sum() if not income_df.empty else 0
    active_days = len(df['日期'].unique())
    
    return {
        'balance': total_income - total_expense,
        'total_expense': total_expense,
        'total_income': total_income,
        'expense_count': len(expense_df),
        'income_count': len(income_df),
        'total_count': len(expense_df) + len(income_df),
        'active_days': active_days,
        'avg_transaction': round(expense_df['金額'].mean(), 2) if len(expense_df) > 0 else 0,
        'avg_daily_expense': round(total_expense / max(1, active_days), 2),
        'expense_ratio': round(total_expense / total_income * 100, 2) if total_income > 0 else 0
    }

@app.route('/api/yearly_data')
def yearly_data():
    try:
        df = load_alipay_data()
        logger.info(f"数据加载完成，总行数: {len(df)}")
        
        # 获取所有可用年份并转换为普通 Python 列表
        available_years = sorted(df['時間'].dt.year.unique().tolist(), reverse=True)
        
        # 从请求参数获取年份，如果没有指定则使用最新的可用年份
        year = request.args.get('year', available_years[0], type=int)
        logger.info(f"请求年度数据: {year}")
        
        current_year_df = df[df['時間'].dt.year == year].copy()
        logger.info(f"当前年份数据行数: {len(current_year_df)}")
        
        last_year_df = df[df['時間'].dt.year == (year - 1)].copy()
        logger.info(f"上一年数据行数: {len(last_year_df)}")
        
        # 获取筛选参数
        filter_type = request.args.get('filter', 'all')
        
        # 根据筛选类型過滤数据
        if filter_type == 'large':
            current_year_df = current_year_df[current_year_df['金額'] >= 1000]  # 修改为1000元
            last_year_df = last_year_df[last_year_df['金額'] >= 1000] if not last_year_df.empty else last_year_df
        elif filter_type == 'small':
            current_year_df = current_year_df[current_year_df['金額'] < 1000]  # 修改为1000元
            last_year_df = last_year_df[last_year_df['金額'] < 1000] if not last_year_df.empty else last_year_df
        
        # 当前年度统计
        current_stats = calculate_yearly_stats(current_year_df)
        
        # 计算环比变化
        yearly_stats = {
            **current_stats,  # 保持原有统计数据
        }
        
        # 只有当存在上一年数据时才计算环比
        if len(last_year_df) > 0:
            print("发现上一年数据，计算环比...")
            last_year_stats = calculate_yearly_stats(last_year_df)
            
            # 计算环比变化率
            balance_change_rate = calculate_change_rate(current_stats['balance'], last_year_stats['balance'])
            expense_change_rate = calculate_change_rate(current_stats['total_expense'], last_year_stats['total_expense'])
            income_change_rate = calculate_change_rate(current_stats['total_income'], last_year_stats['total_income'])
            transaction_change_rate = calculate_change_rate(current_stats['total_count'], last_year_stats['total_count'])
            
            # 添加环比变化数据，确保转换为普通 Python 数值，處理 None 值
            yearly_stats.update({
                'balance_change': float(current_stats['balance'] - last_year_stats['balance']),
                'expense_change': float(current_stats['total_expense'] - last_year_stats['total_expense']),
                'income_change': float(current_stats['total_income'] - last_year_stats['total_income']),
                'transaction_change': int(current_stats['total_count'] - last_year_stats['total_count']),
                'balance_change_rate': float(balance_change_rate) if balance_change_rate is not None else None,
                'expense_change_rate': float(expense_change_rate) if expense_change_rate is not None else None,
                'income_change_rate': float(income_change_rate) if income_change_rate is not None else None,
                'transaction_change_rate': float(transaction_change_rate) if transaction_change_rate is not None else None
            })
            print("环比数据计算完成")
        else:
            print("未找到上一年数据")
            yearly_stats.update({
                'balance_change': None,
                'expense_change': None,
                'income_change': None,
                'transaction_change': None,
                'balance_change_rate': None,
                'expense_change_rate': None,
                'income_change_rate': None,
                'transaction_change_rate': None
            })
        
        # 處理月度数据
        months = sorted(current_year_df['月份'].unique().tolist())
        expenses = []
        incomes = []
        
        for month in months:
            month_data = current_year_df[current_year_df['月份'] == month]
            expenses.append(float(round(month_data[
                (month_data['記錄類型'] == '支出') & 
                (month_data['記錄類型'] != '退款')
            ]['金額'].sum(), 2)))
            incomes.append(float(round(month_data[
                (month_data['記錄類型'] == '收入') & 
                (month_data['記錄類型'] != '退款')
            ]['金額'].sum(), 2)))
        
        # 處理分类数据
        expense_df = current_year_df[
            (current_year_df['記錄類型'] == '支出') & 
            (current_year_df['記錄類型'] != '退款')
        ]
        categories = expense_df.groupby('主類別')['金額'].sum().sort_values(ascending=False)
        
        return jsonify({
            'yearly_stats': yearly_stats,
            'months': months,
            'expenses': expenses,
            'incomes': incomes,
            'categories': categories.index.tolist(),
            'amounts_by_category': [float(x) for x in categories.values.tolist()],
            'available_years': available_years,
            'current_year': int(year)
        })
        
    except Exception as e:
        logger.error(f"處理年度数据时出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def calculate_change_rate(current, previous):
    """计算环比变化率，處理特殊情况"""
    try:
        if previous == 0:
            if current == 0:
                return 0
            return None  # 无法计算增长率（从0增长）
        if current == 0:
            return -100  # 降低100%
        return round((current - previous) / abs(previous) * 100, 2)
    except:
        return None

def calculate_yearly_stats(df):
    """计算年度统计数据"""
    if df.empty:
        return {
            'balance': 0,
            'total_expense': 0,
            'total_income': 0,
            'expense_count': 0,
            'income_count': 0,
            'total_count': 0,
            'active_days': 0,
            'avg_transaction': 0,
            'avg_daily_expense': 0,
            'avg_monthly_income': 0,
            'expense_ratio': 0
        }
    
    # 過滤有效交易
    expense_df = df[
        (df['記錄類型'] == '支出') & 
        (df['記錄類型'] != '退款')
    ]
    income_df = df[
        (df['記錄類型'] == '收入') & 
        (df['記錄類型'] != '退款')
    ]
    
    total_expense = expense_df['金額'].sum() if not expense_df.empty else 0
    total_income = income_df['金額'].sum() if not income_df.empty else 0
    
    return {
        'balance': total_income - total_expense,
        'total_expense': total_expense,
        'total_income': total_income,
        'expense_count': len(expense_df),
        'income_count': len(income_df),
        'total_count': len(expense_df) + len(income_df),
        'active_days': len(df['日期'].unique()),
        'avg_transaction': round(expense_df['金額'].mean(), 2) if len(expense_df) > 0 else 0,
        'avg_daily_expense': round(total_expense / max(1, len(df['日期'].unique())), 2),
        'avg_monthly_income': round(total_income / 12, 2),
        'expense_ratio': round(total_expense / total_income * 100, 2) if total_income > 0 else 0
    }

@app.route('/api/category_analysis')
def category_analysis():
    try:
        df = load_alipay_data()
        category = request.args.get('category')
        time_range = request.args.get('range', 'year')
        year = request.args.get('year')
        month = request.args.get('month')
        min_amount = request.args.get('min_amount')  # 添加金額范围参数
        max_amount = request.args.get('max_amount')  # 添加金額范围参数
        
        print(f"== category is {category} ===")
        # 過滤支出数据
        expense_df = df[
            (df['記錄類型'] == '支出') & 
            (df['記錄類型'] != '退款')
        ]
        
        # 根据时间范围過滤数据
        if time_range == 'year' and year:
            expense_df = expense_df[expense_df['時間'].dt.year == int(year)]
        elif time_range == 'month' and year and month:
            expense_df = expense_df[
                (expense_df['時間'].dt.year == int(year)) & 
                (expense_df['時間'].dt.month == int(month))
            ]
        
        # 按金額范围過滤
        try:
            if min_amount:
                min_val = float(min_amount)
                expense_df = expense_df[expense_df['金額'] >= min_val]
            if max_amount and max_amount.lower() != 'infinity':
                max_val = float(max_amount)
                expense_df = expense_df[expense_df['金額'] < max_val]
        except ValueError as e:
            logger.warning(f"金額转换错误: {str(e)}")
        
        if not category:
            # 返回所有可用分类及其基本统计信息
            categories_stats = expense_df.groupby('主類別').agg({
                '金額': ['sum', 'count', 'mean']
            }).round(2)
            
            categories_stats.columns = ['total', 'count', 'avg']
            categories_stats = categories_stats.sort_values('total', ascending=False)
            
            return jsonify({
                'categories': categories_stats.index.tolist(),
                'stats': {
                    'totals': categories_stats['total'].tolist(),
                    'counts': categories_stats['count'].tolist(),
                    'averages': categories_stats['avg'].tolist()
                }
            })
        
        # 過滤特定分类的数据
        category_df = expense_df[expense_df['主類別'] == category]
        print(f"== category_df is {category_df} ===")
        if category_df.empty:
            return jsonify({
                'error': f'未找到分类 "{category}" 的数据'
            }), 404
        
        # 计算日期范围
        if time_range == 'all':
            date_range = (category_df['時間'].max() - category_df['時間'].min()).days + 1
        elif time_range == 'year':
            date_range = 365
        else:
            date_range = calendar.monthrange(int(year), int(month))[1]
        
        # 基础统计
        total_expense = category_df['金額'].sum()
        transaction_count = len(category_df)
        avg_amount = round(category_df['金額'].mean(), 2) if transaction_count > 0 else 0
        
        # 计算该分类占总支出的比例
        if time_range == 'all':
            total_all_expense = expense_df['金額'].sum()
        elif time_range == 'year':
            total_all_expense = expense_df[expense_df['時間'].dt.year == int(year)]['金額'].sum()
        else:
            total_all_expense = expense_df[
                (expense_df['時間'].dt.year == int(year)) & 
                (expense_df['時間'].dt.month == int(month))
            ]['金額'].sum()
        
        expense_ratio = round((total_expense / total_all_expense * 100), 2) if total_all_expense > 0 else 0
        
        # 按时间分组统计
        if time_range == 'all':
            # 按年份分组
            grouped = category_df.groupby(category_df['時間'].dt.strftime('%Y'))
            total_grouped = expense_df.groupby(expense_df['時間'].dt.strftime('%Y'))
        elif time_range == 'year':
            # 按月份分组
            grouped = category_df.groupby(category_df['時間'].dt.strftime('%Y-%m'))
            total_grouped = expense_df.groupby(expense_df['時間'].dt.strftime('%Y-%m'))
        else:
            # 按日期分组
            grouped = category_df.groupby(category_df['時間'].dt.strftime('%Y-%m-%d'))
            total_grouped = expense_df.groupby(expense_df['時間'].dt.strftime('%Y-%m-%d'))
        
        time_series = grouped['金額'].sum().round(2)
        total_series = total_grouped['金額'].sum().round(2)
        transaction_counts = grouped.size()

        # 计算每個时间点的占比
        ratios = []
        for date in time_series.index:
            if date in total_series.index:
                total = max(abs(total_series[date]), 1e-9)  # 确保是正数，避免除 0
                ratio = (abs(time_series[date]) / total * 100).round(2)  # 用绝对值计算占比
            else:
                ratio = 0
            ratios.append(ratio)
        
        # 计算消费规律
        hour_pattern = category_df.groupby(category_df['時間'].dt.hour)['金額'].agg([
            ('count', 'count'),
            ('sum', 'sum')
        ]).round(2)
        category_df = category_df.copy()  # 显式创建副本
        category_df['金額'] = category_df['金額'].abs()  # 现在不会触发 SettingWithCopyWarning

        # 计算金額分布
        amount_ranges = [0, 50, 100, 200, 500, 1000, float('inf')]
        amount_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
        amount_dist = pd.cut(category_df['金額'], bins=amount_ranges, labels=amount_labels)
        amount_distribution = amount_dist.value_counts().sort_index().astype(float)

        if amount_distribution.sum() > 0:
            percentages = ((amount_distribution / amount_distribution.sum()) * 100).astype(float).round(2).tolist()
        else:
            percentages = [0] * len(amount_distribution)

        return jsonify({
            'category': category,
            'stats': {
                'total_expense': float(total_expense),
                'transaction_count': int(transaction_count),
                'avg_amount': float(avg_amount),
                'expense_ratio': float(expense_ratio),
                'date_range': int(date_range),
                'max_amount': float(category_df['金額'].max()) if not category_df.empty else 0,
                'min_amount': float(category_df['金額'].min()) if not category_df.empty else 0,
                'median_amount': float(category_df['金額'].median()) if not category_df.empty else 0
            },
            'trend': {
                'dates': time_series.index.tolist(),
                'amounts': time_series.values.tolist(),
                'counts': transaction_counts.tolist(),
                'ratios': ratios
            },
            'pattern': {
                'hours': hour_pattern.index.tolist(),
                'counts': hour_pattern['count'].tolist(),
                'amounts': hour_pattern['sum'].tolist(),
                'averages': (hour_pattern['sum'] / hour_pattern['count']).round(2).tolist()
            },
            'distribution': {
                'ranges': amount_distribution.index.tolist(),
                'counts': amount_distribution.values.tolist(),
                'percentages': percentages
            }
        })
        
    except Exception as e:
        logger.error(f"處理分类分析数据时出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/available_dates')
def get_available_dates():
    try:
        df = load_alipay_data()
        dates = df['時間'].dt.strftime('%Y-%m').unique().tolist()
        dates.sort(reverse=True)
        
        return jsonify({
            'success': True,
            'months': dates,
            'years': sorted(df['時間'].dt.year.unique().tolist(), reverse=True)
        })
    except Exception as e:
        logger.error(f"Error getting available dates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/category_available_dates')
def get_category_available_dates():
    try:
        df = load_alipay_data()
        
        # 按年份分组获取每年的可用月份
        dates = pd.DataFrame({
            'year': df['時間'].dt.year,
            'month': df['時間'].dt.month
        })
        
        available_months = {}
        # 修改这里：按年份倒序排列
        for year in sorted(dates['year'].unique(), reverse=True):  # 添加 reverse=True
            months = sorted(dates[dates['year'] == year]['month'].unique())
            available_months[int(year)] = [int(m) for m in months]
        
        return jsonify({
            'years': sorted(dates['year'].unique().tolist(), reverse=True),  # 添加 reverse=True
            'months': available_months
        })
    except Exception as e:
        logger.error(f"Error getting category available dates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/settings')
def settings():
    return render_template('settings.html', active_page='settings')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Starting file upload...")
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上傳'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': '不支持的文件类型'}), 400
            
        # 确保目录存在
        session_dir = get_session_dir()
        if not os.path.exists(session_dir):
            os.makedirs(session_dir, mode=0o700)
            
        # 安全地保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(session_dir, filename)
        file.save(filepath)
        
        # 更新会话时间戳
        session['created_at'] = datetime.now().timestamp()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': '文件上傳成功'
        })
        
    except Exception as e:
        logger.exception("Upload failed with error:")  # 这会记录完整的堆栈跟踪
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files')
def list_files():
    """列出当前会话的文件"""
    session_dir = get_session_dir()
    files = []
    if os.path.exists(session_dir):
        for filename in os.listdir(session_dir):
            if filename.endswith('.csv'):
                files.append({
                    'name': filename,
                    'size': os.path.getsize(os.path.join(session_dir, filename))
                })
    return jsonify({'files': files})

@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    """删除会话中的文件"""
    if not filename.endswith('.csv'):
        return jsonify({'success': False, 'error': '无效的文件名'})
    
    try:
        session_dir = get_session_dir()
        filepath = os.path.join(session_dir, secure_filename(filename))
        
        if os.path.exists(filepath):
            # os.remove(filepath)
            # 清除数据缓存
            # load_alipay_data.cache_clear()
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': '文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/session/status')
def get_session_status():
    """获取会话状态"""
    if 'user_id' not in session:
        return jsonify({
            'active': False,
            'message': '会话未初始化'
        })
    
    # 不再检查超时，只返回会话状态
    return jsonify({
        'active': True,
        'message': '会话活跃'
    })

@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    """手动清除用户数据"""
    try:
        if 'user_id' in session:
            session_dir = get_session_dir()
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
            session.clear()
        return jsonify({'success': True, 'message': '数据已清除'})
    except Exception as e:
        logger.error(f"清除数据时出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'清除数据失败: {str(e)}'
        }), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """只在会话過期时清理数据"""
    try:
        if 'session_start' in session:
            start_time = datetime.strptime(session['session_start'], '%Y-%m-%d %H:%M:%S')
            expire_time = start_time + timedelta(minutes=30)  # 改为30分钟
            
            if datetime.now() >= expire_time:
                if 'user_id' in session:
                    session_dir = get_session_dir()
                    if os.path.exists(session_dir):
                        shutil.rmtree(session_dir)
                    session.clear()
                return jsonify({'success': True, 'message': '会话已過期,数据已清理'})
            
        return jsonify({'success': True, 'message': '会话未過期'})
        
    except Exception as e:
        logger.error(f"清理数据时出错: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

# 添加应用关闭时的清理
@atexit.register
def cleanup_all():
    """应用关闭时清理所有临时数据"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, mode=0o700)
    except Exception as e:
        logger.error(f"清理临时文件夹失败: {str(e)}")

# 确保临时文件根目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 添加获取会话剩余时间的接口
@app.route('/api/session/time_remaining')
def get_session_time_remaining():
    if 'session_start' not in session:
        logger.info("No session_start found in session")
        return jsonify({'remaining': 0})
    
    start_time = datetime.strptime(session['session_start'], '%Y-%m-%d %H:%M:%S')
    expire_time = start_time + timedelta(minutes=30)
    now = datetime.now()
    
    if now >= expire_time:
        try:
            session_dir = get_session_dir()
            logger.info(f"Session expired. Checking directory: {session_dir}")
            
            if os.path.exists(session_dir):
                # 检查目录是否正在使用
                try:
                    # 尝试创建一個临时文件来测试目录是否可写
                    test_file = os.path.join(session_dir, '.test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    # os.remove(test_file)
                    
                    # 如果可以写入，说明目录没有被锁定
                    files = os.listdir(session_dir)
                    logger.info(f"Found files: {files}")
                    
                    # 逐個检查并删除文件
                    for file in files:
                        filepath = os.path.join(session_dir, file)
                        try:
                            # 尝试打开文件，确保没有其他进程在使用
                            with open(filepath, 'r') as f:
                                pass
                            # 如果成功打开，则删除文件
                            # os.remove(filepath)
                            logger.info(f"Deleted file: {file}")
                        except IOError as e:
                            logger.warning(f"Cannot delete file {file}: {str(e)}")
                            continue
                    
                    # 最后删除目录
                    os.rmdir(session_dir)
                    logger.info("Directory deleted successfully")
                    
                except IOError as e:
                    logger.warning(f"Directory is in use: {str(e)}")
                    # 如果目录被锁定，不进行删除
                    return jsonify({'remaining': 30, 'extended': True})
            
            # 清除缓存和会话
            load_alipay_data.cache_clear()
            session.clear()
            logger.info("Cache and session cleared")
            
            return jsonify({'remaining': 0, 'expired': True})
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            # 如果清理過程出错，给用户多一些时间
            return jsonify({'remaining': 30, 'error': str(e), 'extended': True})
    
    remaining_seconds = int((expire_time - now).total_seconds())
    logger.debug(f"Session remaining time: {remaining_seconds} seconds")
    return jsonify({'remaining': remaining_seconds, 'expired': False})

@app.route('/api/available_years')
def get_available_years():
    """获取数据中所有可用的年份"""
    try:
        df = load_alipay_data()

        # 确保 '日期' 列是 datetime 类型
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

        # 检查是否有转换失败的情况（即 NaT 值）
        if df['日期'].isna().any():
            raise ValueError("日期列包含无法解析的值")

        # 提取所有不重复的年份
        years = sorted(df['日期'].dt.year.unique().tolist(), reverse=True)

        return jsonify({
            'success': True,
            'years': years
        })

    except Exception as e:
        logger.error(f"获取可用年份时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取可用年份失败: {str(e)}'
        }), 500


def analyze_merchants(df):
    """商家消费分析"""
    # 只分析支出数据
    expense_df = df[df['記錄類型'] == '支出']
    
    # 按商家分组统计
    merchant_stats = expense_df.groupby('商家').agg({
        '金額': ['count', 'sum', 'mean'],
        '時間': lambda x: (x.max() - x.min()).days + 1  # 交易跨度
    }).round(2)
    
    merchant_stats.columns = ['交易次数', '总金額', '平均金額', '交易跨度']
    
    # 识别常客商家(最近3個月有2次以上消费)
    recent_date = df['時間'].max()
    three_months_ago = recent_date - pd.Timedelta(days=90)
    recent_df = expense_df[expense_df['時間'] >= three_months_ago]
    
    frequent_merchants = []
    for merchant, group in recent_df.groupby('商家'):
        if len(group) >= 2:
            frequent_merchants.append({
                'name': merchant,
                'amount': group['金額'].sum(),
                'count': len(group),
                'last_visit': group['時間'].max().strftime('%Y-%m-%d')
            })
    
    # 按消费金額排序
    frequent_merchants.sort(key=lambda x: x['amount'], reverse=True)
    
    return {
        'merchant_stats': merchant_stats.to_dict('index'),
        'frequent_merchants': frequent_merchants[:10]  # 只返回前10個常客商家
    }

def analyze_scenarios(df):
    """消费场景分析"""
    # 创建副本避免 SettingWithCopyWarning
    expense_df = df[df['記錄類型'] == '支出'].copy()
    
    # 1. 线上/线下消费
    online_keywords = [
        '淘宝', '天猫', '京东', '拼多多', '美团', '饿了么', 'App Store', 'Steam', 
        'Apple Music', 'iCloud', '网易', '支付宝', '微信', '闲鱼', '得物'
    ]
    expense_df.loc[:, '消费场景'] = expense_df['商家'].apply(
        lambda x: '线上' if any(k in str(x) for k in online_keywords) else '线下'
    )
    
    # 2. 消费时段分析
    expense_df.loc[:, '消费时段'] = expense_df['時間'].dt.hour.map(
        lambda x: '清晨(6-9点)' if 6 <= x < 9
        else '上午(9-12点)' if 9 <= x < 12
        else '中午(12-14点)' if 12 <= x < 14
        else '下午(14-17点)' if 14 <= x < 17
        else '傍晚(17-20点)' if 17 <= x < 20
        else '晚上(20-23点)' if 20 <= x < 23
        else '深夜(23-6点)'
    )
    
    # 3. 消费金額层级
    expense_df.loc[:, '消费层级'] = expense_df['金額'].map(
        lambda x: '大额(1000+)' if x >= 1000
        else '中额(300-1000)' if x >= 300
        else '小额(100-300)' if x >= 100
        else '零花(0-100)'
    )
    
    # 汇总各维度的统计数据
    scenario_stats = []
    
    # 添加线上/线下统计
    online_stats = expense_df.groupby('消费场景')['金額'].sum()
    for scene, amount in online_stats.items():
        scenario_stats.append({
            'name': scene,
            'value': float(amount),
            'category': '渠道'
        })
    
    # 添加时段统计
    time_stats = expense_df.groupby('消费时段')['金額'].sum()
    for period, amount in time_stats.items():
        scenario_stats.append({
            'name': period,
            'value': float(amount),
            'category': '时段'
        })
    
    # 添加金額层级统计
    level_stats = expense_df.groupby('消费层级')['金額'].sum()
    for level, amount in level_stats.items():
        scenario_stats.append({
            'name': level,
            'value': abs(float(amount)),
            'category': '层级'
        })
    
    return scenario_stats

def analyze_habits(df):
    """消费习惯分析"""
    expense_df = df[df['記錄類型'] == '支出'].copy()
    
    # 1. 基础统计
    daily_expenses = expense_df.groupby(expense_df['時間'].dt.date)['金額'].sum()
    daily_avg = float(daily_expenses.mean())
    active_days = int(len(daily_expenses))
    
    # 2. 计算周末消费比例
    weekend_expenses = expense_df[expense_df['時間'].dt.dayofweek.isin([5, 6])]['金額'].sum()
    weekend_ratio = float((weekend_expenses / expense_df['金額'].sum() * 100))
    
    # 3. 计算固定支出比例
    monthly_recurring = expense_df.groupby(['商家', expense_df['時間'].dt.month]).size()
    recurring_merchants = monthly_recurring[monthly_recurring >= 2].index.get_level_values(0).unique()
    fixed_expenses = expense_df[expense_df['商家'].isin(recurring_merchants)]['金額'].sum()
    fixed_ratio = float((fixed_expenses / expense_df['金額'].sum() * 100))
    
    # 4. 计算月初消费比例
    month_start = expense_df[expense_df['時間'].dt.day <= 5]['金額'].sum()
    month_start_ratio = float((month_start / expense_df['金額'].sum() * 100))
    
    return {
        'daily_avg': round(daily_avg, 2),
        'active_days': active_days,
        'weekend_ratio': round(weekend_ratio, 1),
        'fixed_expenses': round(fixed_ratio, 1),
        'month_start_ratio': round(month_start_ratio, 1)
    }

def generate_smart_tags(df):
    """智能标签生成"""
    expense_df = df[df['記錄類型'] == '支出']
    result = {
        'tags': [],
        'time_pattern': '',
        'spending_preference': '',
        'spending_pattern': '',
        'spending_power': ''
    }
    
    # 计算总体消费情况
    total_expense = expense_df['金額'].sum()
    avg_expense = expense_df['金額'].mean()
    daily_expense = expense_df.groupby(expense_df['時間'].dt.date)['金額'].sum().mean()
    
    # 时间模式分析
    hour_stats = expense_df.groupby(expense_df['時間'].dt.hour).size()
    peak_hours = hour_stats[hour_stats > hour_stats.mean()].index.tolist()
    
    if 22 in peak_hours or 23 in peak_hours:
        result['tags'].append('夜间消费达人')
        result['time_pattern'] = '您偏好在夜间消费，要注意作息哦'
    elif 6 in peak_hours or 7 in peak_hours:
        result['tags'].append('早起达人')
        result['time_pattern'] = '您是個早起消费的生活达人'
    else:
        result['time_pattern'] = '您的消费时间比较规律，集中在日间'
    
    # 消费偏好分析
    category_ratio = expense_df.groupby('主類別')['金額'].sum() / total_expense
    top_categories = category_ratio[category_ratio > 0.15].index.tolist()
    
    preference_desc = []
    for category in top_categories:
        result['tags'].append(f'{category}控')
        preference_desc.append(f'{category}({(category_ratio[category]*100):.1f}%)')
    
    result['spending_preference'] = f"最常消费的品类是{', '.join(preference_desc)}"
    
    # 消费规律分析
    daily_expenses = expense_df.groupby(expense_df['時間'].dt.date)['金額'].sum()
    cv = daily_expenses.std() / daily_expenses.mean()
    
    if cv < 0.5:
        result['tags'].append('消费稳健派')
        result['spending_pattern'] = '您的消费非常有规律，是個理性消费者'
    elif cv < 0.8:
        result['tags'].append('平衡消费派')
        result['spending_pattern'] = '您的消费较为均衡，适度有波动'
    else:
        result['tags'].append('随性消费派')
        result['spending_pattern'] = '您的消费比较随性，可能需要更多预算管理'
    
    # 消费能力分析
    if daily_expense > 500:
        result['tags'].append('高消费人群')
        result['spending_power'] = f'日均消费{daily_expense:.0f}元，属于高消费人群'
    elif daily_expense > 200:
        result['tags'].append('中等消费人群')
        result['spending_power'] = f'日均消费{daily_expense:.0f}元，消费能力适中'
    else:
        result['tags'].append('理性消费人群')
        result['spending_power'] = f'日均消费{daily_expense:.0f}元，消费比较节制'
    
    return result

def analyze_payment_methods(df):
    """分析支付方式的使用情况"""
    expense_df = df[df['記錄類型'] == '支出'].copy()
    
    # 清理和标准化支付方式名称
    def standardize_payment_method(method):
        method = str(method).split('(')[0].strip()
        # 统一支付方式名称
        if '余额宝' in method or '红包' in method:
            return '余额宝'
        return method
    
    expense_df['支付方式'] = expense_df['帳戶'].apply(standardize_payment_method)
    
    # 基础统计
    payment_stats = expense_df.groupby('支付方式').agg({
        '金額': ['count', 'sum', 'mean'],
        '時間': lambda x: x.dt.date.nunique()  # 使用天数
    })
    
    # 重命名列
    payment_stats.columns = ['交易次数', '總金額', '平均金額', '使用天数']
    
    # 计算使用频率和金額占比
    total_amount = float(expense_df['金額'].sum())  # 转换为 float
    total_count = int(len(expense_df))  # 转换为 int
    
    # 转换为列表格式
    payment_list = []
    for method, stats in payment_stats.iterrows():
        payment_list.append({
            'name': method,
            'transaction_count': int(stats['交易次数']),  # 转换为 int
            'total_amount': float(stats['總金額']),  # 转换为 float
            'avg_amount': float(stats['平均金額']),  # 转换为 float
            'usage_days': int(stats['使用天数']),  # 转换为 int
            'amount_ratio': float(stats['總金額'] / total_amount * 100),  # 转换为 float
            'count_ratio': float(stats['交易次数'] / total_count * 100)  # 转换为 float
        })
    
    # 按总金額排序
    payment_list.sort(key=lambda x: x['total_amount'], reverse=True)
    
    return payment_list

@app.route('/api/yearly_analysis')
def yearly_analysis():
    try:
        df = load_alipay_data()

        # 确保 '時間' 列是 datetime 类型
        df['時間'] = pd.to_datetime(df['時間'], errors='coerce')

        # 检查是否有 NaT 值（即转换失败的情况）
        if df['時間'].isna().any():
            raise ValueError("時間列包含无法解析的值")

        year = request.args.get('year', type=int)
        min_amount = request.args.get('min_amount', type=float)
        max_amount = request.args.get('max_amount', type=float)

        # 现在可以安全地使用 .dt 访问器
        current_year_df = df[df['時間'].dt.year == year] if year else df

        last_year = year - 1 if year else df['時間'].dt.year.max() - 1
        last_year_df = df[df['時間'].dt.year == last_year]

        # 应用金额筛选
        if min_amount:
            current_year_df = current_year_df[current_year_df['金額'] >= min_amount]
            last_year_df = last_year_df[last_year_df['金額'] >= min_amount]
        if max_amount:
            current_year_df = current_year_df[current_year_df['金額'] < max_amount]
            last_year_df = last_year_df[last_year_df['金額'] < max_amount]

        # 過滤有效交易（排除退款）
        current_expense_df = current_year_df[
            (current_year_df['記錄類型'] == '支出') & 
            (current_year_df['記錄類型'] != '退款')
        ]
        current_income_df = current_year_df[
            (current_year_df['記錄類型'] == '收入') & 
            (current_year_df['記錄類型'] != '退款')
        ]

        last_expense_df = last_year_df[
            (last_year_df['記錄類型'] == '支出') & 
            (last_year_df['記錄類型'] != '退款')
        ]
        last_income_df = last_year_df[
            (last_year_df['記錄類型'] == '收入') & 
            (last_year_df['記錄類型'] != '退款')
        ]

        # 计算数据
        current_expense = current_expense_df['金額'].sum()
        current_income = current_income_df['金額'].sum()
        current_balance = current_income - current_expense

        last_expense = last_expense_df['金額'].sum()
        last_income = last_income_df['金額'].sum()
        last_balance = last_income - last_expense

        # 生成月份列表
        all_months = [f"{year}-{str(month).zfill(2)}" for month in range(1, 13)]

        # 按月统计
        monthly_expenses = current_expense_df.groupby(
            current_expense_df['時間'].dt.strftime('%Y-%m')
        )['金額'].sum().reindex(all_months, fill_value=0)

        monthly_incomes = current_income_df.groupby(
            current_income_df['時間'].dt.strftime('%Y-%m')
        )['金額'].sum().reindex(all_months, fill_value=0)

        # 分类统计
        category_expenses = current_expense_df.groupby('主類別')['金額'].sum()
        category_incomes = current_income_df.groupby('主類別')['金額'].sum()

        # 年度统计
        yearly_stats = {
            'balance': float(current_balance),
            'total_expense': float(current_expense),
            'total_income': float(current_income),
            'expense_count': int(len(current_expense_df)),
            'income_count': int(len(current_income_df)),
            'total_count': int(len(current_expense_df) + len(current_income_df)),
            'active_days': int(len(current_year_df['時間'].dt.date.unique())),
            'avg_transaction': float(current_expense_df['金額'].mean()) if len(current_expense_df) > 0 else 0,
            'avg_daily_expense': float(current_expense / max(1, len(current_year_df['時間'].dt.date.unique()))),
            'avg_monthly_income': float(current_income / 12),
            'expense_ratio': float(current_expense / current_income * 100) if current_income > 0 else 0,
            'comparisons': {
                'balance': {
                    'change': float(current_balance - last_balance) if len(last_year_df) > 0 else None,
                    'rate': float((current_balance - last_balance) / abs(last_balance) * 100) if len(last_year_df) > 0 and last_balance != 0 else None
                },
                'expense': {
                    'change': float(current_expense - last_expense) if len(last_year_df) > 0 else None,
                    'rate': float((current_expense - last_expense) / last_expense * 100) if len(last_year_df) > 0 and last_expense != 0 else None
                },
                'income': {
                    'change': float(current_income - last_income) if len(last_year_df) > 0 else None,
                    'rate': float((current_income - last_income) / last_income * 100) if len(last_year_df) > 0 and last_income != 0 else None
                },
                'count': {
                    'change': int(len(current_expense_df) + len(current_income_df) - len(last_expense_df) - len(last_income_df)) if len(last_year_df) > 0 else None,
                    'rate': float((len(current_expense_df) + len(current_income_df) - len(last_expense_df) - len(last_income_df)) / (len(last_expense_df) + len(last_income_df)) * 100) if len(last_year_df) > 0 and (len(last_expense_df) + len(last_income_df)) != 0 else None
                }
            }
        }

        # 记录日志
        logger.info(f"Yearly stats: income={current_income}, expense={current_expense}, balance={current_balance}")

        return jsonify({
            'success': True,
            'data': {
                'trends': {
                    'months': monthly_expenses.index.tolist(),
                    'expenses': monthly_expenses.values.tolist(),
                    'incomes': monthly_incomes.values.tolist()
                },
                'categories': {
                    'expense': {
                        'names': category_expenses.index.tolist(),
                        'amounts': category_expenses.values.tolist()
                    },
                    'income': {
                        'names': category_incomes.index.tolist(),
                        'amounts': category_incomes.values.tolist()
                    }
                },
                'yearly_stats': yearly_stats
            }
        })

    except Exception as e:
        logger.error(f"Error in yearly analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.before_request
def check_session_expiry():
    if 'created_at' in session:
        session_age = datetime.now().timestamp() - session['created_at']
        if session_age > 1800:  # 30分钟過期
            session.clear()
            return redirect(url_for('index'))

@app.teardown_request
def cleanup_session(exception=None):
    """请求结束时检查并清理過期的会话文件"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        current_time = datetime.now().timestamp()
        
        # 遍历所有会话目录
        for user_dir in os.listdir(upload_folder):
            dir_path = os.path.join(upload_folder, user_dir)
            if os.path.isdir(dir_path):
                dir_stat = os.stat(dir_path)
                # 如果目录超過30分钟未修改，则删除
                if current_time - dir_stat.st_mtime > 1800:
                    shutil.rmtree(dir_path)
    except Exception as e:
        logger.error(f"Session cleanup error: {str(e)}")

# 在应用启动时确保上傳目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 确保目录权限正确
def ensure_upload_dir():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, mode=0o700)
    else:
        os.chmod(UPLOAD_FOLDER, 0o700)

# 在应用启动时调用
ensure_upload_dir()

if __name__ == '__main__':
    # 判断是否在生产环境
    is_production = os.environ.get('PRODUCTION', False)
    
    if is_production:
        # 生产环境配置
        app.config['TEMPLATES_AUTO_RELOAD'] = False
        app.run(
            host='0.0.0.0',
            port=8080,
            debug=False,
            use_reloader=False
        )
    else:
        # 开发环境配置
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        extra_files = ['app.py']
        app.run(
            host='0.0.0.0',
            port=8080,
            debug=True,
            use_reloader=True,
            extra_files=extra_files,
            reloader_interval=2
        )
