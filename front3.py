import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import json
import numpy as np
import yfinance as yf
from forex_python.converter import CurrencyRates
import hashlib
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shutil
from datetime import datetime, timedelta
import plotly.graph_objects as go
# Add missing import for BytesIO and time
from io import BytesIO
import time

def init_db():
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  email TEXT,
                  life_stage TEXT,
                  goals TEXT,
                  income REAL,
                  expenses TEXT,
                  savings REAL,
                  risk_profile TEXT,
                  notifications TEXT,
                  currency TEXT,
                  created_date TEXT,
                  last_login TEXT,
                  settings TEXT,
                  reset_token TEXT,
                  reset_token_expiry TEXT)''')
    
    # Create the transactions table with all required columns
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            category TEXT,
            name TEXT,
            amount REAL,
            interest_rate REAL,
            minimum_payment REAL,
            transaction_type TEXT,
            date TEXT,
            description TEXT,
            payment_method TEXT,
            recurring BOOLEAN,
            tags TEXT,
            related_debt_id INTEGER,
            subcategory TEXT,
            FOREIGN KEY(related_debt_id) REFERENCES transactions(id)
        )
    """)
    
    conn.commit()
    conn.close()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  symbol TEXT,
                  quantity REAL,
                  purchase_price REAL,
                  purchase_date TEXT,
                  portfolio_type TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    c.execute('''CREATE TABLE IF NOT EXISTS bills
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  name TEXT,
                  amount REAL,
                  due_date TEXT,
                  frequency TEXT,
                  category TEXT,
                  auto_pay BOOLEAN,
                  reminder_days INTEGER,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  report_type TEXT,
                  report_date TEXT,
                  report_data TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    conn.commit()
    conn.close()

class UserSettings:
    def __init__(self, username):
        self.username = username
        self.settings = self._load_settings()

    def _default_settings(self):
        return {
            'currency': 'USD',
            'date_format': '%Y-%m-%d',
            'theme': 'light',
            'notifications': {
                'email': True,
                'push': False,
                'bill_reminders': True,
                'budget_alerts': True,
                'investment_alerts': True
            },
            'budget_alerts': {
                'threshold': 80,
                'frequency': 'weekly'
            },
            'report_preferences': {
                'frequency': 'monthly',
                'include_categories': ['all'],
                'export_format': 'pdf'
            },
            'dashboard_widgets': [
                'budget_overview',
                'recent_transactions',
                'upcoming_bills',
                'investment_summary',
                'savings_goals'
            ]
        }

    def _load_settings(self):
        try:
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            c.execute("SELECT settings FROM users WHERE username=?", (self.username,))
            result = c.fetchone()
            conn.close()
            
            if result and result[0]:
                settings = json.loads(result[0])
                settings.setdefault('currency', 'GHS')
                settings.setdefault('theme', 'light')
                settings.setdefault('notifications', {
                    'email': True,
                    'bill_reminders': True
                })
                return settings
            return self._default_settings()
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
            return self._default_settings()
        
    def save_settings(self, settings):
        try:
            if 'goals' in settings and not isinstance(settings['goals'], str):
                settings['goals'] = json.dumps(settings['goals'])
                
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            c.execute("""
                UPDATE users 
                SET settings=? 
                WHERE username=?
            """, (json.dumps(settings), self.username))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")
            return False

class FinancialAnalyzer:
    def __init__(self, username):
        self.username = username
        
    def calculate_net_worth_details(self):
        try:
            conn = sqlite3.connect('financial_planner.db')
            
            # Get assets (all income transactions)
            assets_query = """
                SELECT 
                    category,
                    SUM(amount) as total
                FROM transactions 
                WHERE username=? 
                    AND transaction_type='Income'
                    AND date >= date('now', '-30 days')
                GROUP BY category
            """
            assets_df = pd.read_sql_query(assets_query, conn, params=(self.username,))
            
            # Get liabilities (all expense and debt transactions)
            liabilities_query = """
                SELECT 
                    category,
                    SUM(amount) as total
                FROM transactions 
                WHERE username=? 
                    AND (transaction_type='Expense' OR category='debt')
                    AND date >= date('now', '-30 days')
                GROUP BY category
            """
            liabilities_df = pd.read_sql_query(liabilities_query, conn, params=(self.username,))
            
            # Get portfolio value
            portfolio_query = """
                SELECT 
                    symbol,
                    SUM(quantity * purchase_price) as total
                FROM portfolios 
                WHERE username=?
                GROUP BY symbol
            """
            portfolio_df = pd.read_sql_query(portfolio_query, conn, params=(self.username,))
            
            conn.close()
            
            # Calculate total assets including portfolio value
            total_assets = assets_df['total'].sum() + portfolio_df['total'].sum()
            total_liabilities = liabilities_df['total'].sum()
            
            # Add portfolio to assets dataframe
            if not portfolio_df.empty:
                portfolio_row = pd.DataFrame({
                    'category': ['Investments'],
                    'total': [portfolio_df['total'].sum()]
                })
                assets_df = pd.concat([assets_df, portfolio_row], ignore_index=True)
            
            return {
                'assets': assets_df if not assets_df.empty else pd.DataFrame({'category': [], 'total': []}),
                'liabilities': liabilities_df if not liabilities_df.empty else pd.DataFrame({'category': [], 'total': []}),
                'net_worth': total_assets - total_liabilities
            }
        except Exception as e:
            st.error(f"Error calculating net worth: {str(e)}")
            return {
                'assets': pd.DataFrame({'category': [], 'total': []}),
                'liabilities': pd.DataFrame({'category': [], 'total': []}),
                'net_worth': 0
            }
    
    
    
    def analyze_spending_patterns(self):
        conn = sqlite3.connect('financial_planner.db')
        current_month = datetime.now().strftime('%Y-%m')
        income_df = pd.read_sql_query("""
            SELECT SUM(amount) as income
            FROM transactions 
            WHERE username=? 
            AND transaction_type='Income'
            AND strftime('%Y-%m', date)=?
        """, conn, params=(self.username, current_month))
        expenses_df = pd.read_sql_query("""
            SELECT SUM(amount) as expenses
            FROM transactions 
            WHERE username=? 
            AND transaction_type='Expense'
            AND strftime('%Y-%m', date)=?
        """, conn, params=(self.username, current_month))
        conn.close()
        total_income = income_df['income'].iloc[0] or 0
        total_expenses = expenses_df['expenses'].iloc[0] or 0
        monthly_savings = total_income - total_expenses
        return {
            'monthly_savings': monthly_savings,
            'total_income': total_income,
            'total_expenses': total_expenses
        }

class InvestmentAnalyzer:
    def __init__(self, username):
        self.username = username
        
    def get_portfolio_performance(self):
        try:
            conn = sqlite3.connect('financial_planner.db')
            portfolio = pd.read_sql_query("""
                SELECT * FROM portfolios WHERE username=?
            """, conn, params=(self.username,))
            conn.close()
            
            if portfolio.empty:
                return {
                    'total_value': 0,
                    'portfolio_return': 0.0,
                    'portfolio_risk': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            total_value = 0
            returns = []
            
            for _, position in portfolio.iterrows():
                try:
                    if position['portfolio_type'] == 'stock':
                        ticker = yf.Ticker(position['symbol'])
                        history = ticker.history(period='1d')
                        if not history.empty:
                            current_price = history['Close'].iloc[-1]
                        else:
                            current_price = position['purchase_price']
                    else:
                        current_price = position['purchase_price']
                    
                    position_value = current_price * position['quantity']
                    position_return = (current_price - position['purchase_price']) / position['purchase_price']
                    total_value += position_value
                    returns.append(position_return)
                except Exception as e:
                    st.warning(f"Error processing {position['symbol']}: {str(e)}")
                    total_value += position['purchase_price'] * position['quantity']
                    returns.append(0.0)
            
            portfolio_return = np.mean(returns) if returns else 0.0
            portfolio_risk = np.std(returns) if returns else 0.0
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return {
                'total_value': total_value,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio
            }
        except Exception as e:
            st.error(f"Error calculating portfolio performance: {str(e)}")
            return None
class DebtManager:
    def __init__(self, username):
        self.username = username
    
    def add_debt(self, name, amount, interest_rate, minimum_payment):
        try:
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            c.execute("""
                INSERT INTO transactions 
                (username, category, description, amount, interest_rate, 
                minimum_payment, transaction_type, date)
                VALUES (?, 'debt', ?, ?, ?, ?, 'Expense', datetime('now'))
            """, (self.username, name, amount, interest_rate, minimum_payment))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error adding debt: {str(e)}")
            return False
        
    def analyze_debt(self):
        conn = sqlite3.connect('financial_planner.db')
        try:
            debts = pd.read_sql_query("""
                SELECT amount, interest_rate 
                FROM transactions 
                WHERE username=? AND category='debt'
            """, conn, params=(self.username,))
            
            total_debt = debts['amount'].sum()
            avg_interest = debts['interest_rate'].mean() if not debts.empty else None
            
            return {
                'total_debt': total_debt,
                'average_interest': avg_interest
            }
        except Exception as e:
            st.error(f"Error analyzing debt: {str(e)}")
            return None
        finally:
            conn.close()

    def make_payment(self, debt_id, payment_amount):
        try:
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            
            # First get the current debt amount
            c.execute("""
                SELECT amount FROM transactions 
                WHERE id=? AND username=? AND category='debt'
            """, (debt_id, self.username))
            result = c.fetchone()
            
            if not result:
                raise ValueError("Debt not found")
                
            current_amount = result[0]
            
            # Calculate new amount after payment
            new_amount = current_amount - payment_amount
            
            # Update the debt amount
            c.execute("""
                UPDATE transactions 
                SET amount=? 
                WHERE id=? AND username=? AND category='debt'
            """, (new_amount, debt_id, self.username))
            
            # Record the payment as a new transaction
            try:
                c.execute("""
                    INSERT INTO transactions 
                    (username, category, description, amount, transaction_type, date, related_debt_id)
                    VALUES (?, 'debt_payment', 'Debt Payment', ?, 'Expense', datetime('now'), ?)
                """, (self.username, payment_amount, debt_id))
            except sqlite3.OperationalError:
                # If related_debt_id column doesn't exist, insert without it
                c.execute("""
                    INSERT INTO transactions 
                    (username, category, description, amount, transaction_type, date)
                    VALUES (?, 'debt_payment', 'Debt Payment', ?, 'Expense', datetime('now'))
                """, (self.username, payment_amount))
            
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error processing payment: {str(e)}")
            return False
        finally:
            conn.close()
    def get_debt_details(self):
        conn = sqlite3.connect('financial_planner.db')
        try:
            # Modified query to handle missing columns and use description as name
            query = """
                SELECT 
                    id,
                    category,
                    description as name,  -- Use description as name
                    amount,
                    COALESCE(interest_rate, 0.0) as interest_rate,
                    COALESCE(minimum_payment, 0.0) as minimum_payment,
                    date as start_date
                FROM transactions 
                WHERE username=? 
                AND category='debt' 
                AND amount > 0
                ORDER BY interest_rate DESC
            """
            
            debts = pd.read_sql_query(query, conn, params=(self.username,))
            

            
            # Ensure all required columns exist with default values
            if 'minimum_payment' not in debts.columns:
                debts['minimum_payment'] = 0.0
            if 'interest_rate' not in debts.columns:
                debts['interest_rate'] = 0.0
            if 'name' not in debts.columns:
                debts['name'] = 'Unnamed Debt'
                
            debts['minimum_payment'] = 0.0
                
            return debts
        except Exception as e:
            st.error(f"Error getting debt details: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
 

    def display_debt_management_ui(self):
        st.header("Debt Management (Instalments)")
        
        with st.expander("Add New Debt"):
            col1, col2 = st.columns(2)
            with col1:
                debt_name = st.text_input("Debt Name", key="new_debt_name")
                debt_amount = st.number_input("Amount", min_value=0.0, key="new_debt_amount")
            with col2:
                interest_rate = st.number_input("Interest Rate (%)", 
                                            min_value=0.0, 
                                            key="new_debt_interest")
                min_payment = st.number_input("Minimum Monthly Payment", 
                                            min_value=0.0, 
                                            key="new_debt_min_payment")
            
            if st.button("Add Debt", key="add_debt_button"):
                if self.add_debt(debt_name, debt_amount, interest_rate, min_payment):
                    st.success("Debt added successfully!")
                    st.rerun()

        # Modify the get_debt_details query
        debts = self.get_debt_details()
        if not debts.empty:
            st.subheader("Your Debts")
            for idx, debt in debts.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{debt['name']}**")
                        st.write(f"Balance: ${debt['amount']:,.2f}")
                    with col2:
                        st.write(f"Interest: {debt['interest_rate']}%")
                        st.write(f"Min Payment: ${debt['minimum_payment']:,.2f}")
                    with col3:
                        payment_amount = st.number_input(
                            "Payment Amount",
                            min_value=0.0,
                            max_value=float(debt['amount']),
                            key=f"payment_{debt['id']}"
                        )
                        if st.button("Make Payment", key=f"pay_{debt['id']}"):
                            if self.make_payment(debt['id'], payment_amount):
                                st.success("Payment processed successfully!")
                                st.rerun()
                    st.divider()
        else:
            st.info("You have no active debts.")

class BillManager:
    def __init__(self, username):
        self.username = username
    
    def get_upcoming_bills(self, days=30):
        conn = sqlite3.connect('financial_planner.db')
        current_date = datetime.now().date()
        end_date = current_date + timedelta(days=days)
        bills = pd.read_sql_query("""
            SELECT * FROM bills 
            WHERE username=? AND due_date BETWEEN ? AND ?
        """, conn, params=(self.username, current_date, end_date))
        conn.close()
        return bills.sort_values('due_date')

class ReportGenerator:
    def __init__(self, username):
        self.username = username
    
    def generate_monthly_report(self, month=None):
        if month is None:
            month = datetime.now().strftime('%Y-%m')
        analyzer = FinancialAnalyzer(self.username)
        investment_analyzer = InvestmentAnalyzer(self.username)
        income_expense = self._analyze_income_expenses(month)
        budget_performance = self._analyze_budget_performance(month)
        investment_performance = investment_analyzer.get_portfolio_performance()
        savings_progress = self._analyze_savings_progress()
        report = {
            'month': month,
            'income_expense': income_expense,
            'budget_performance': budget_performance,
            'investment_performance': investment_performance,
            'savings_progress': savings_progress,
            'recommendations': self._generate_recommendations()
        }
        self._save_report(report, 'monthly')
        return report
    
    def _analyze_income_expenses(self, month):
        pass
    
    def _analyze_budget_performance(self, month):
        pass
    
    def _analyze_savings_progress(self):
        pass
    
    def _generate_recommendations(self):
        pass
    
    def _save_report(self, report_data, report_type):
        conn = sqlite3.connect('financial_planner.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO reports (username, report_type, report_date, report_data)
            VALUES (?, ?, ?, ?)
        """, (self.username, report_type, datetime.now().strftime('%Y-%m-%d'),
              json.dumps(report_data)))
        conn.commit()
        conn.close()
        
# DebtManager class is missing analyze_debt method


def reset_password(token, new_password):
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    c.execute("""
        SELECT username FROM users 
        WHERE reset_token=? AND reset_token_expiry > ?
    """, (token, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    result = c.fetchone()
    
    if result:
        username = result[0]
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        c.execute("""
            UPDATE users 
            SET password=?, reset_token=NULL, reset_token_expiry=NULL 
            WHERE username=?
        """, (hashed_password, username))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def send_reset_email(email, reset_link):
    sender_email = "your-email@domain.com"
    sender_password = "your-app-specific-password"
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = email
    message["Subject"] = "Password Reset Request"
    
    body = f"""
    Hello,
    
    You have requested to reset your password. Please click the link below to reset your password:
    
    {reset_link}
    
    This link will expire in 1 hour.
    
    If you did not request this reset, please ignore this email.
    
    Best regards,
    Financial Life Planner Team
    """
    
    message.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def validate_password(password):
    # Add return message for better error handling
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search("[a-z]", password):
        return False, "Password must contain lowercase letters"
    if not re.search("[A-Z]", password):
        return False, "Password must contain uppercase letters"
    if not re.search("[0-9]", password):
        return False, "Password must contain numbers"
    return True, "Password is valid"

def check_session_timeout():
    if 'last_activity' in st.session_state:
        if (datetime.now() - st.session_state.last_activity).total_seconds() > 1800:
            st.session_state.authenticated = False
            return True
    st.session_state.last_activity = datetime.now()
    return False

def create_spending_trend_chart(data):
    data['date'] = pd.to_datetime(data['date'])
    data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
    spending_by_date = data.groupby(['date', 'category'])['amount'].sum().reset_index()
    fig = px.line(spending_by_date, 
                  x='date', 
                  y='amount',
                  color='category',
                  title='Spending Trends Over Time')
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        hovermode='x unified'
    )
    return fig

def create_portfolio_pie_chart(data):
    fig = px.pie(data, values='value', names='asset',
                 title='Portfolio Asset Allocation')
    return fig

def export_data(data, format='csv'):
    if format == 'csv':
        return data.to_csv(index=False)
    elif format == 'excel':
        return data.to_excel(index=False)
    elif format == 'pdf':
        pass

def check_budget_alerts(username):
    conn = sqlite3.connect('financial_planner.db')
    current_month = datetime.now().strftime('%Y-%m')
    spending = pd.read_sql_query("""
        SELECT category, SUM(amount) as spent
        FROM transactions 
        WHERE username=? AND strftime('%Y-%m', date)=?
        GROUP BY category
    """, conn, params=(username, current_month))
    conn.close()
    user_settings = UserSettings(username)
    budget = user_settings.settings.get('budget', {})
    alerts = []
    for _, row in spending.iterrows():
        if row['category'] in budget:
            budget_amount = budget[row['category']]
            if row['spent'] > budget_amount * 0.8:
                alerts.append(f"Warning: {row['category']} spending at {(row['spent']/budget_amount)*100:.1f}% of budget")
    return alerts

def track_goal_progress(goal):
    current_amount = calculate_current_amount(goal)
    target_amount = goal['target_amount']
    deadline = datetime.strptime(goal['deadline'], '%Y-%m-%d')
    days_remaining = (deadline - datetime.now()).days
    progress = {
        'percentage': (current_amount / target_amount) * 100,
        'remaining_amount': target_amount - current_amount,
        'days_remaining': days_remaining,
        'on_track': is_goal_on_track(current_amount, target_amount, days_remaining)
    }
    return progress

def calculate_current_amount(goal):
    # Add implementation for calculating current amount for a goal
    try:
        conn = sqlite3.connect('financial_planner.db')
        result = pd.read_sql_query("""
            SELECT SUM(amount) as total
            FROM transactions 
            WHERE category=? AND username=?
        """, conn, params=(goal['name'], st.session_state.username))
        return float(result['total'].iloc[0]) if not result.empty else 0
    except Exception as e:
        st.error(f"Error calculating current amount: {str(e)}")
        return 0
    finally:
        conn.close()

def is_goal_on_track(current_amount, target_amount, days_remaining):
    if days_remaining <= 0:
        return current_amount >= target_amount
    daily_target = (target_amount - current_amount) / days_remaining
    return daily_target > 0



def convert_currency(amount, from_currency, to_currency):
    c = CurrencyRates()
    try:
        rate = c.get_rate(from_currency, to_currency)
        return amount * rate
    except Exception as e:
        st.error(f"Currency conversion error: {str(e)}")
        return amount

def backup_database():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'backup_{timestamp}.db'
    try:
        shutil.copy2('financial_planner.db', backup_file)
        return True
    except Exception as e:
        st.error(f"Backup error: {str(e)}")
        return False

def manage_categories():
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS custom_categories
                 (username TEXT,
                  category_name TEXT,
                  category_type TEXT,
                  PRIMARY KEY (username, category_name))''')
    conn.commit()
    conn.close()

st.set_page_config(
    page_title="Financial Planner",
    page_icon="💰",
    layout="wide",
)

def get_spending_data(username):
    try:
        # Use safe_db_query for better error handling
        query = """
            SELECT 
                date,  -- Get the full datetime string
                category,
                SUM(amount) as amount
            FROM transactions 
            WHERE username=? 
                AND transaction_type='Expense'
                AND date >= date('now', '-30 days')
                AND category IS NOT NULL
                AND category != ''
            GROUP BY date, category
            ORDER BY date
        """
        spending_data = safe_db_query(query, params=(username,))

        if not spending_data.empty:
            # Convert date using a more flexible approach
            spending_data['date'] = pd.to_datetime(spending_data['date'], format='mixed')
            
            # Get user's currency settings
            user_settings = UserSettings(username)
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)
            
            # Create the spending trend chart
            fig = px.line(
                spending_data, 
                x='date', 
                y='amount',
                color='category',
                title='Spending Trends Over Time'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=f"Amount ({currency_symbol})",
                hovermode='x unified',
                legend_title="Categories",
                showlegend=True,
                height=400  # Fixed height for better display
            )
            
            # Improve trace appearance
            fig.update_traces(
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=6)
            )
            
            return fig
        else:
            return None
            
    except Exception as e:
        st.error(f"Error retrieving spending data: {str(e)}")
        return None
      
def get_currency_symbol(currency_code):
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "GHS": "₵"
    }
    return currency_symbols.get(currency_code, currency_code)

def convert_amount(amount, from_currency, to_currency):
    if from_currency == to_currency:
        return amount
    try:
        c = CurrencyRates()
        rate = c.get_rate(from_currency, to_currency)
        return amount * rate
    except Exception as e:
        st.warning(f"Could not convert currency: {e}")
        return amount
    
def create_net_worth_chart(assets_df, liabilities_df, currency_symbol):
    # Create net worth waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Net Worth",
        orientation="v",
        measure=["relative"] * len(assets_df) + ["relative"] * len(liabilities_df) + ["total"],
        x=list(assets_df['category']) + list(liabilities_df['category']) + ['Net Worth'],
        textposition="outside",
        text=[f"{currency_symbol}{x:,.0f}" for x in assets_df['total']] +
            [f"{currency_symbol}{x:,.0f}" for x in liabilities_df['total']] +
            [f"{currency_symbol}{(assets_df['total'].sum() - liabilities_df['total'].sum()):,.0f}"],
        y=list(assets_df['total']) + list(-liabilities_df['total']) + 
        [assets_df['total'].sum() - liabilities_df['total'].sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Net Worth Breakdown",
        showlegend=True,
        height=400
    )
    return fig




def create_spending_pie_chart(spending_data, currency_symbol):
    fig = px.pie(spending_data, 
                values='amount', 
                names='category',
                title='Spending Distribution by Category',
                hole=0.4)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(
        annotations=[dict(text=f'Total\n{currency_symbol}{spending_data["amount"].sum():,.0f}', 
                        x=0.5, y=0.5, font_size=12, showarrow=False)]
    )
    return fig

def create_income_expense_trend(data, currency_symbol):
    fig = px.line(data, 
                x='date', 
                y=['income', 'expenses'],
                title='Income vs Expenses Trend',
                labels={'value': f'Amount ({currency_symbol})', 'date': 'Date'})
    fig.update_layout(hovermode='x unified')
    return fig

# Add context manager for database connections
def get_db_connection():
    return sqlite3.connect('financial_planner.db')

# Use in functions like:
def some_database_operation():
    with get_db_connection() as conn:
        c = conn.cursor()
        try:
            # database operations
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise e

def safe_db_query(query, params=None):
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except sqlite3.Error as e:
        st.error(f"Query execution error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
        
def create_dashboard_charts(username, currency_symbol):
    # Get assets data
    assets_query = """
        SELECT category, SUM(amount) as total
        FROM transactions 
        WHERE username=? AND transaction_type='Income'
        GROUP BY category
    """
    assets_df = safe_db_query(assets_query, params=(username,))
    
    # Get liabilities data
    liabilities_query = """
        SELECT category, SUM(amount) as total
        FROM transactions 
        WHERE username=? AND transaction_type='Expense'
        AND category LIKE '%debt%'
        GROUP BY category
    """
    liabilities_df = safe_db_query(liabilities_query, params=(username,))
    
    # Get spending data
    spending_query = """
        SELECT category, SUM(amount) as amount 
        FROM transactions 
        WHERE username=? AND transaction_type='Expense'
        AND date >= date('now', '-30 days')
        GROUP BY category
    """
    spending_df = safe_db_query(spending_query, params=(username,))
    
    # Get trend data
    trend_query = """
        SELECT 
            date,
            SUM(CASE WHEN transaction_type='Income' THEN amount ELSE 0 END) as income,
            SUM(CASE WHEN transaction_type='Expense' THEN amount ELSE 0 END) as expenses
        FROM transactions 
        WHERE username=? AND date >= date('now', '-90 days')
        GROUP BY date
        ORDER BY date
    """
    trend_df = safe_db_query(trend_query, params=(username,))
    
    # Create charts only if data exists
    charts = {}
    
    if not assets_df.empty or not liabilities_df.empty:
        charts['net_worth'] = create_net_worth_chart(
            assets_df if not assets_df.empty else pd.DataFrame({'category': [], 'total': []}),
            liabilities_df if not liabilities_df.empty else pd.DataFrame({'category': [], 'total': []}),
            currency_symbol
        )
    
    if not spending_df.empty:
        charts['spending'] = create_spending_pie_chart(spending_df, currency_symbol)
    
    if not trend_df.empty:
        charts['trend'] = create_income_expense_trend(trend_df, currency_symbol)
    
    return charts

def safe_db_query(query, params=None):
    conn = None
    try:
        conn = sqlite3.connect('financial_planner.db')
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def convert_amount(amount, from_currency, to_currency):
    if from_currency == to_currency:
        return amount
    try:
        c = CurrencyRates()
        rate = c.get_rate(from_currency, to_currency)
        return amount * rate
    except Exception as e:
        st.warning(f"Could not convert currency: {e}")
        return amount  # Return original amount if conversion fails

def get_theme_colors():
    return {
        "light": {
            "background": "#FFFFFF",
            "text": "#000000",
            "sidebar_bg": "#F0F2F6",
            "sidebar_text": "#111111",
            "accent": "#FF4B4B",
            "secondary": "#F0F2F6",
            "success": "#00C853",
            "warning": "#FFA726",
            "error": "#FF4B4B",
            "info": "#2196F3",
            "card_bg": "#FFFFFF",
            "card_border": "#E0E0E0",
            "input_bg": "#FFFFFF",
            "input_text": "#000000",
            "input_border": "#E0E0E0",
            "hover": "#F5F5F5"
        },
        "dark": {
            "background": "#262730",
            "text": "#FFFFFF",
            "sidebar_bg": "#1E1E1E",
            "sidebar_text": "#FFFFFF",
            "accent": "#FF4B4B",
            "secondary": "#3B3B3B",
            "success": "#4CAF50",
            "warning": "#FFA726",
            "error": "#FF5252",
            "info": "#2196F3",
            "card_bg": "#1E1E1E",
            "card_border": "#333333",
            "input_bg": "#333333",
            "input_text": "#FFFFFF",
            "input_border": "#444444",
            "hover": "#363636"
        },
        "blue": {
            "background": "#1E3D59",
            "text": "#FFFFFF",
            "sidebar_bg": "#152D43",
            "sidebar_text": "#E0E0E0",
            "accent": "#17C3B2",
            "secondary": "#2B5876",
            "success": "#00BFA5",
            "warning": "#FFD54F",
            "error": "#FF5252",
            "info": "#40C4FF",
            "card_bg": "#234B6E",
            "card_border": "#1A364F",
            "input_bg": "#2B5876",
            "input_text": "#FFFFFF",
            "input_border": "#17C3B2",
            "hover": "#2B5876"
        },
        "green": {
            "background": "#2C5530",
            "text": "#FFFFFF",
            "sidebar_bg": "#1E3B22",
            "sidebar_text": "#E0E0E0",
            "accent": "#8BC34A",
            "secondary": "#3E7B45",
            "success": "#66BB6A",
            "warning": "#FDD835",
            "error": "#FF5252",
            "info": "#29B6F6",
            "card_bg": "#2F5934",
            "card_border": "#1E3B22",
            "input_bg": "#3E7B45",
            "input_text": "#FFFFFF",
            "input_border": "#8BC34A",
            "hover": "#3E7B45"
        },
        "purple": {
            "background": "#46344E",
            "text": "#FFFFFF",
            "sidebar_bg": "#2A1F2F",
            "sidebar_text": "#E0E0E0",
            "accent": "#9C27B0",
            "secondary": "#5D4266",
            "success": "#7CB342",
            "warning": "#FFB300",
            "error": "#FF5252",
            "info": "#29B6F6",
            "card_bg": "#533A5A",
            "card_border": "#2A1F2F",
            "input_bg": "#5D4266",
            "input_text": "#FFFFFF",
            "input_border": "#9C27B0",
            "hover": "#5D4266"
        },
        "pink": {
            "background": "#FFE4E1",
            "text": "#4A4A4A",
            "sidebar_bg": "#FFB6C1",
            "sidebar_text": "#4A4A4A",
            "accent": "#FF69B4",
            "secondary": "#FFC0CB",
            "success": "#66BB6A",
            "warning": "#FFA726",
            "error": "#FF4081",
            "info": "#29B6F6",
            "card_bg": "#FFF0F5",
            "card_border": "#FFB6C1",
            "input_bg": "#FFFFFF",
            "input_text": "#4A4A4A",
            "input_border": "#FF69B4",
            "hover": "#FFC0CB"
        }
    }

def get_nav_bar_style(colors):
    return f"""
        <style>
            .nav-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem 2rem;
                background-color: {colors['sidebar_bg']};
                border-bottom: 1px solid {colors['card_border']};
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
            }}
            
            .nav-brand {{
                font-size: 1.5rem;
                font-weight: bold;
                color: {colors['sidebar_text']};
                text-decoration: none;
            }}
            
            .nav-menu {{
                display: flex;
                gap: 1rem;
                align-items: center;
            }}
            
            .nav-item {{
                padding: 0.5rem 1rem;
                color: {colors['sidebar_text']};
                text-decoration: none;
                border-radius: 0.375rem;
                transition: all 0.2s;
                cursor: pointer;
            }}
            
            .nav-item:hover {{
                background-color: {colors['hover']};
            }}
            
            .nav-item.active {{
                background-color: {colors['accent']};
                color: white;
            }}
            
            .user-menu {{
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            
            .user-info {{
                color: {colors['sidebar_text']};
            }}
            
            .main-content {{
                margin-top: 4rem;
                padding: 1rem;
            }}
            
            .dropdown {{
                position: relative;
                display: inline-block;
            }}
            
            .dropdown-content {{
                display: none;
                position: absolute;
                top: 100%;
                right: 0;
                background-color: {colors['card_bg']};
                min-width: 160px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                border-radius: 0.375rem;
                z-index: 1001;
            }}
            
            .dropdown:hover .dropdown-content {{
                display: block;
            }}
            
            .dropdown-item {{
                padding: 0.75rem 1rem;
                color: {colors['text']};
                text-decoration: none;
                display: block;
            }}
            
            .dropdown-item:hover {{
                background-color: {colors['hover']};
            }}
        </style>
    """


# Enhanced Streamlit Interface
def main():
    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = datetime.now()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "currency" not in st.session_state:
        st.session_state.currency = "GHS"  # Default currency

    # Initialize user settings if authenticated
    if st.session_state.authenticated:
        user_settings = UserSettings(st.session_state.username)

    # Check session timeout
    if st.session_state.authenticated:
        if check_session_timeout():
            st.warning("Session expired. Please login again.")
            st.rerun()

        # Set currency symbol globally
        st.session_state.currency = user_settings.settings.get('currency', 'GHS')

        # Initialize theme and settings
        theme = user_settings.settings.get('theme', 'light')
        theme_colors = get_theme_colors()

        if theme in theme_colors:
            colors = theme_colors[theme]
            st.markdown(f"""
                <style>
                    .nav-menu {{
                        margin-top:30px
                    }}

                    .stApp {{
                        background-color: {colors['background']};
                        color: {colors['text']};
                    }}
                    [data-testid="stSidebar"] {{
                        background-color: {colors['sidebar_bg']} !important;
                        color: {colors['sidebar_text']};
                        border-right: 1px solid {colors['card_border']};
                    }}
                    .stButton > button {{
                        background-color: {colors['accent']};
                        color: white;
                        border: none;
                        border-radius: 0.375rem;
                        padding: 0.5rem 1rem;
                        transition: all 0.3s ease;
                    }}
                    .stButton > button:hover {{
                        background-color: {colors['hover']};
                        transform: translateY(-1px);
                    }}
                </style>
            """, unsafe_allow_html=True)

    # Login/Registration Section
    if not st.session_state.authenticated:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")

            with st.expander("Forgot Password?"):
                forgot_username = st.text_input("Enter your username", key="forgot_username")
                forgot_email = st.text_input("Enter your registered email", key="forgot_email")
                if st.button("Reset Password"):
                    try:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            SELECT email FROM users 
                            WHERE username=? AND email=?
                        """, (forgot_username, forgot_email))
                        result = c.fetchone()
                        if result:
                            reset_token = hashlib.sha256(
                                f"{forgot_username}{datetime.now().strftime('%Y-%m-%d-%H')}".encode()
                            ).hexdigest()
                            c.execute("""
                                UPDATE users 
                                SET reset_token=?, reset_token_expiry=?
                                WHERE username=?
                            """, (reset_token, 
                                (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                                forgot_username))
                            conn.commit()
                            reset_link = f"http://yourdomain.com/reset_password?token={reset_token}"
                            st.success("Password reset link has been generated!")
                            st.code(reset_link)
                            st.info("In a production environment, this link would be sent to your email.")
                        else:
                            st.error("Username and email combination not found.")
                    except Exception as e:
                        st.error(f"Error resetting password: {str(e)}")
                    finally:
                        conn.close()

            if st.button("Login"):
                try:
                    conn = sqlite3.connect('financial_planner.db')
                    c = conn.cursor()
                    c.execute("SELECT password FROM users WHERE username=?", (login_username,))
                    result = c.fetchone()
                    if result and result[0] == hashlib.sha256(login_password.encode()).hexdigest():
                        st.session_state.username = login_username
                        st.session_state.authenticated = True
                        st.session_state.current_page = "Dashboard"
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                except Exception as e:
                    st.error(f"Error during login: {str(e)}")
                finally:
                    conn.close()

        with col2:
            st.subheader("Register")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_email = st.text_input("Email")

            if st.button("Register"):
                if not reg_username or not reg_password or not reg_email:
                    st.error("Please fill in all fields")
                else:
                    is_valid, message = validate_password(reg_password)
                    if is_valid:
                        try:
                            conn = sqlite3.connect('financial_planner.db')
                            c = conn.cursor()
                            hashed_password = hashlib.sha256(reg_password.encode()).hexdigest()
                            c.execute("""
                                INSERT INTO users (username, password, email, created_date)
                                VALUES (?, ?, ?, ?)
                            """, (reg_username, hashed_password, reg_email, datetime.now().strftime('%Y-%m-%d')))
                            conn.commit()
                            st.success("Registration successful! Please login.")
                        except sqlite3.IntegrityError:
                            st.error("Username already exists")
                        except Exception as e:
                            st.error(f"Error during registration: {str(e)}")
                        finally:
                            conn.close()
                    else:
                        st.error(message)

    # Navigation and Page Content
    if st.session_state.authenticated:
        # Initialize managers and analyzers
        user_settings = UserSettings(st.session_state.username)
        financial_analyzer = FinancialAnalyzer(st.session_state.username)
        investment_analyzer = InvestmentAnalyzer(st.session_state.username)
        debt_manager = DebtManager(st.session_state.username)
        bill_manager = BillManager(st.session_state.username)
        report_generator = ReportGenerator(st.session_state.username)

        # Get theme colors
        theme = user_settings.settings.get('theme', 'light')
        colors = get_theme_colors()[theme]

        # Apply navigation bar styles
        st.markdown(get_nav_bar_style(colors), unsafe_allow_html=True)

        # Create navigation bar using buttons
        if st.session_state.authenticated:
            st.sidebar.title("Navigation")
            pages = ["Dashboard", "Transactions", "Budget Tracking", "Budget Planner", "Investments", 
                    "Goals", "Debt Management", "Bills & Subscriptions", "Reports & Analysis", "Settings"]
            
            selected_page = st.sidebar.selectbox(
                "Go to",
                pages,
                index=pages.index(st.session_state.current_page)
            )
            
            if selected_page != st.session_state.current_page:
                st.session_state.current_page = selected_page
                # Update query parameters
                st.query_params["page"] = selected_page
                st.rerun()
        
        

        # Get the selected page from URL parameters
        query_params = st.query_params
        
        # Handle logout first
        if 'logout' in query_params:
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        # Get the page from query parameters or use default
        selected_page = query_params.get('page', ['Dashboard'])[0]

        # Use selected_page instead of page in your conditional statements
        if st.session_state.current_page == "Dashboard":
            st.header("Financial Dashboard")
            
            try:
                net_worth_details = financial_analyzer.calculate_net_worth_details()
                portfolio_performance = investment_analyzer.get_portfolio_performance()
            
                # In the Dashboard section
                currency = user_settings.settings.get('currency', 'GHS')
                currency_symbol = get_currency_symbol(currency)
                net_worth_details = financial_analyzer.calculate_net_worth_details()
                
                # Add this to display alerts
                alerts = check_budget_alerts(st.session_state.username)
                if alerts:
                    st.subheader("Budget Alerts")
                    for alert in alerts:
                        st.warning(alert)
                        


                # Quick metrics
                col1, col2, col3, col4 = st.columns(4)
                # In the Dashboard section
                if 'last_net_worth' not in st.session_state:
                    st.session_state.last_net_worth = 0

                with col1:
                    current_net_worth = net_worth_details['net_worth']
                    st.metric(
                        "Net Worth", 
                        f"{currency_symbol}{net_worth_details['net_worth']:,.2f}",
                        delta=None
                    )
                    # Update the last net worth for next time
                    st.session_state.last_net_worth = current_net_worth
        
                    # Create and display net worth waterfall chart
                    net_worth_chart = create_net_worth_chart(
                        net_worth_details['assets'],
                        net_worth_details['liabilities'],
                        currency_symbol
                    )
                
                
        
                # In the dashboard section
                if 'username' not in st.session_state:
                    st.warning("Please log in to view your dashboard.")
                else:
                    charts = create_dashboard_charts(st.session_state.username, currency_symbol)
                    
                    if charts:
                        if 'net_worth' in charts:
                            st.plotly_chart(charts['net_worth'], use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            try:
                                spending_chart = get_spending_data(st.session_state.username)
                                if spending_chart:
                                    st.plotly_chart(spending_chart, use_container_width=True)
                                else:
                                    st.info("No spending data available for the last 30 days.")
                            except Exception as e:
                                st.error(f"Error displaying spending chart: {str(e)}")
                        
                        with col2:
                            if 'trend' in charts:
                                st.plotly_chart(charts['trend'], use_container_width=True)
                            else:
                                st.info("No trend data available.")
                    else:
                        st.info("Start tracking your finances by adding some transactions!")

                        
                with col2:
                    portfolio = investment_analyzer.get_portfolio_performance()
                    if portfolio_performance:
                        st.metric(
                            "Portfolio Value", 
                            f"{currency_symbol}{portfolio_performance['total_value']:,.2f}",
                            f"{portfolio_performance['portfolio_return']*100:.1f}%"
                        )
                    else:
                        st.metric("Portfolio Value", f"{currency_symbol}0.00")

                with col3:
                    upcoming_bills = bill_manager.get_upcoming_bills(7)
                    if not upcoming_bills.empty:
                        st.metric("Upcoming Bills", f"{currency_symbol}{upcoming_bills['amount'].sum():,.2f}")
                    else:
                        st.metric("Upcoming Bills", f"{currency_symbol}0.00")

                with col4:
                    savings_data = financial_analyzer.analyze_spending_patterns()
                    if savings_data:
                        savings_amount = savings_data['monthly_savings']
                        st.metric("Savings This Month", 
                                f"{currency_symbol}{savings_amount:,.2f}",
                                delta="Income exceeds expenses" if savings_amount > 0 else "Overspending detected")
                    else:
                        st.metric("Savings This Month", f"{currency_symbol}0.00")
                        
            except Exception as e:
                st.error(f"Error loading dashboard metrics: {str(e)}")

            # In the Dashboard section, replace the existing spending data query with:
            conn = sqlite3.connect('financial_planner.db')
            spending_data = pd.read_sql_query("""
                SELECT 
                    date,
                    category,
                    SUM(amount) as amount
                FROM transactions 
                WHERE username=? 
                    AND transaction_type='Expense'
                    AND date >= date('now', '-30 days')
                    AND category IS NOT NULL
                    AND category != ''
                GROUP BY date, category
                ORDER BY date
            """, conn, params=(st.session_state.username,))
            conn.close()

            if not spending_data.empty:
                # Convert date to datetime if it's not already
                spending_data['date'] = pd.to_datetime(spending_data['date'])
                
                # Create the spending trend chart with improved configuration
                fig = px.line(spending_data, 
                            x='date', 
                            y='amount',
                            color='category',
                            title='Spending Trends Over Time')
                
                # Update layout for better readability and interaction
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=f"Amount ({currency_symbol})",
                    hovermode='x unified',
                    legend_title="Categories",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                # Update traces for better visibility
                fig.update_traces(
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No transaction data available for the chart. Add some transactions to see spending trends.")

            # Make sure to close the connection after use
            conn.close()

            # Display recent transactions
            st.subheader("Recent Transactions")
            conn = sqlite3.connect('financial_planner.db')
            transactions = pd.read_sql_query("""
                SELECT date, category, subcategory, amount, description 
                FROM transactions 
                WHERE username=? 
                ORDER BY date DESC LIMIT 10
            """, conn, params=(st.session_state.username,))
            conn.close()

            if not transactions.empty:
                st.dataframe(transactions, use_container_width=True)
            else:
                st.write("No transactions to display.")

        elif st.session_state.current_page == "Budget Planner":
            st.header("Budget Planner")
            
            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            # Income Section
            st.subheader("Monthly Income")
            income_sources = ["Salary", "Investments", "Side Business", "Other"]
            income = {}
            total_income = 0

            for source in income_sources:
                current_income = user_settings.settings.get('income', {}).get(source, 0.0)
                income[source] = st.number_input(f"{source} Income", 
                                            min_value=0.0, 
                                            value=float(current_income),
                                            step=100.0)
                total_income += income[source]

            st.metric("Total Monthly Income", f"{currency_symbol}{total_income:,.2f}")

            # Expense Budget Section
            st.subheader("Monthly Expenses Budget")
            categories = ["Housing", "Utilities", "Food", "Transportation", 
                        "Entertainment", "Savings", "Insurance", "Others"]
            budget = {}
            total_budget = 0

            for category in categories:
                current_budget = user_settings.settings.get('budget', {}).get(category, 0.0)
                budget[category] = st.number_input(f"{category} Budget", 
                                                min_value=0.0, 
                                                value=float(current_budget),
                                                step=10.0)
                total_budget += budget[category]

            st.metric("Total Monthly Budget", f"{currency_symbol}{total_budget:,.2f}")

            # Budget Summary
            st.subheader("Budget Summary")
            remaining = total_income - total_budget
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Monthly Surplus/Deficit", 
                        f"{currency_symbol}{remaining:,.2f}",
                        delta=f"{(remaining/total_income)*100:.1f}% of income" if total_income > 0 else "0%")

            if st.button("Save Budget"):
                user_settings.settings['income'] = income
                user_settings.settings['budget'] = budget
                conn = sqlite3.connect('financial_planner.db')
                c = conn.cursor()
                c.execute("""
                    UPDATE users 
                    SET settings=? 
                    WHERE username=?
                """, (json.dumps(user_settings.settings), st.session_state.username))
                conn.commit()
                conn.close()
                st.success("Budget saved successfully!")

        # Add other pages as needed...
        elif st.session_state.current_page == "Transactions":
            st.header("Transaction Management")

            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            # Add New Transaction
            st.subheader("Add New Transaction")

            # Transaction Type Selection
            transaction_type = st.radio("Transaction Type", ["Expense", "Income"])

            col1, col2 = st.columns(2)

            with col1:
                transaction_date = st.date_input("Date", value=datetime.now())
                # Modify categories based on transaction type
                if transaction_type == "Expense":
                    transaction_category = st.selectbox("Category", 
                        ["Housing", "Utilities", "Food", "Transportation", 
                        "Entertainment", "Savings", "Insurance", "Others"])
                else:
                    transaction_category = st.selectbox("Category", 
                        ["Salary", "Investments", "Business", "Freelance", 
                        "Rental", "Other Income"])
                transaction_amount = st.number_input("Amount", min_value=0.0, step=10.0)

            with col2:
                transaction_subcategory = st.text_input("Subcategory (optional)")
                transaction_description = st.text_input("Description")
                payment_method = st.selectbox("Payment Method", 
                    ["Cash", "Credit Card", "Debit Card", "Bank Transfer", "Other"])

            recurring = st.checkbox("Recurring Transaction")
            tags = st.text_input("Tags (comma-separated)")

            if st.button("Add Transaction"):
                try:
                    conn = sqlite3.connect('financial_planner.db')
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO transactions 
                        (username, date, category, subcategory, amount, description, 
                        payment_method, recurring, tags, transaction_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (st.session_state.username, transaction_date.strftime('%Y-%m-%d'),
                        transaction_category, transaction_subcategory, transaction_amount,
                        transaction_description, payment_method, recurring, tags, transaction_type))
                    conn.commit()
                    conn.close()
                    st.success("Transaction added successfully!")
                except Exception as e:
                    st.error(f"Error adding transaction: {str(e)}")

            # Transaction History
            st.subheader("Transaction History")

            # Filters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                transaction_type_filter = st.multiselect("Transaction Type",
                    ["Income", "Expense"], default=["Income", "Expense"])
            with col2:
                filter_category = st.multiselect("Filter by Category",
                    ["Housing", "Utilities", "Food", "Transportation", 
                    "Entertainment", "Savings", "Insurance", "Others",
                    "Salary", "Investments", "Business", "Freelance", 
                    "Rental", "Other Income"])
            with col3:
                date_range = st.date_input("Date Range", 
                                        value=(datetime.now() - timedelta(days=30), datetime.now()))
            with col4:
                sort_order = st.selectbox("Sort by", ["Date (Latest)", "Date (Oldest)", 
                                                    "Amount (Highest)", "Amount (Lowest)"])

            query = """
                SELECT date, transaction_type, category, subcategory, amount, description, 
                payment_method, tags
                FROM transactions 
                WHERE username=?
            """
            params = [st.session_state.username]

            # Apply filters based on transaction_type
            if transaction_type_filter:
                query += " AND transaction_type IN ({})".format(','.join(['?'] * len(transaction_type_filter)))
                params.extend(transaction_type_filter)

            # Apply filters based on category
            if filter_category:
                query += " AND category IN ({})".format(','.join(['?'] * len(filter_category)))
                params.extend(filter_category)

            conn = sqlite3.connect('financial_planner.db')
            transactions = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if not transactions.empty:
                st.subheader("Transaction List")
                for index, row in transactions.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    with col1:
                        st.write(f"Date: {row['date']}")
                    with col2:
                        st.write(f"Type: {row['transaction_type']}")
                    with col3:
                        st.write(f"Category: {row['category']}")
                    with col4:
                        st.write(f"Amount: {currency_symbol}{abs(row['amount']):,.2f}")
                    with col5:
                        if st.button("Delete", key=f"del_trans_{index}"):
                            try:
                                conn = sqlite3.connect('financial_planner.db')
                                c = conn.cursor()
                                c.execute("""
                                    DELETE FROM transactions 
                                    WHERE username=? AND date=? AND category=? AND amount=? AND description=?
                                """, (st.session_state.username, row['date'], row['category'], 
                                    row['amount'], row['description']))
                                conn.commit()
                                conn.close()
                                st.success("Transaction deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting transaction: {str(e)}")

        elif st.session_state.current_page == "Budget Tracking":
            st.header("Budget Tracking")
            
            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            # Get current month's transactions
            conn = sqlite3.connect('financial_planner.db')
            current_month = datetime.now().strftime('%Y-%m')
            transactions = pd.read_sql_query("""
                    SELECT category, SUM(amount) as spent
                    FROM transactions 
                    WHERE username=? AND strftime('%Y-%m', date)=?
                    GROUP BY category
                """, conn, params=(st.session_state.username, current_month))

            # Get budget data
            budget_data = user_settings.settings.get('budget', {})

            # Create comparison dataframe
            budget_comparison = pd.DataFrame(list(budget_data.items()), columns=['category', 'budget'])
            budget_comparison = budget_comparison.merge(transactions, on='category', how='left')
            budget_comparison['spent'] = budget_comparison['spent'].fillna(0)
            budget_comparison['remaining'] = budget_comparison['budget'] - budget_comparison['spent']
            budget_comparison['percentage'] = (budget_comparison['spent'] / budget_comparison['budget'] * 100).round(1)

            # Display budget progress
            for _, row in budget_comparison.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Ensure progress value is between 0 and 100, then convert to 0-1 range
                    progress = max(0, min(100, abs(row['percentage']))) / 100
                    st.progress(progress)
                with col2:
                    st.write(f"{row['category']}: {currency_symbol}{abs(row['spent']):,.2f} / ${row['budget']:,.2f}")

        elif st.session_state.current_page == "Goals":
            st.header("Financial Goals")
            
            
            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            try:
                # Ensure goals are properly loaded as JSON
                goals_str = user_settings.settings.get('goals', '[]')
                if not isinstance(goals_str, str):
                    goals_str = '[]'
                goals = json.loads(goals_str)

                # Add new goal
                st.subheader("Add New Goal")
                goal_name = st.text_input("Goal Name")
                goal_amount = st.number_input("Target Amount", min_value=0.0, step=100.0)
                goal_deadline = st.date_input("Target Date")

                if st.button("Add Goal"):
                    if goal_name and goal_amount > 0:  # Basic validation
                        new_goal = {
                            "name": goal_name,
                            "amount": float(goal_amount),  # Ensure amount is float
                            "deadline": goal_deadline.strftime("%Y-%m-%d"),
                            "created_date": datetime.now().strftime("%Y-%m-%d")
                        }

                        # Initialize goals list if empty
                        if not isinstance(goals, list):
                            goals = []

                        goals.append(new_goal)

                        # Update settings with new goals
                        user_settings.settings['goals'] = json.dumps(goals)

                        # Save to database using the save_settings method
                        if user_settings.save_settings(st.session_state.username, user_settings.settings):
                            st.success("Goal added successfully!")
                        else:
                            st.error("Failed to save goal")
                    else:
                        st.warning("Please enter a goal name and amount greater than 0")

                # Display existing goals
                if goals and isinstance(goals, list):
                    st.subheader("Your Goals")
                    for i, goal in enumerate(goals):
                        try:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**{goal.get('name', 'Unnamed Goal')}**")
                            with col2:
                                st.write(f"Target: {currency_symbol}{float(goal.get('amount', 0)):,.2f}")
                            with col3:
                                st.write(f"Deadline: {goal.get('deadline', 'No deadline')}")
                        except (KeyError, ValueError, TypeError) as e:
                            st.error(f"Error displaying goal {i+1}: {str(e)}")
                            continue

            except json.JSONDecodeError as e:
                st.error(f"Error loading goals: {str(e)}")
                user_settings.settings['goals'] = '[]'
                if user_settings.save_settings(user_settings.settings):
                    st.info("Goals have been reset. Please try adding a new goal.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

                if goals and isinstance(goals, list):
                    st.subheader("Your Goals")
                    for i, goal in enumerate(goals):
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                        with col1:
                            st.write(f"**{goal.get('name', 'Unnamed Goal')}**")
                        with col2:
                            st.write(f"Target: ${float(goal.get('amount', 0)):,.2f}")
                        with col3:
                            st.write(f"Deadline: {goal.get('deadline', 'No deadline')}")
                        with col4:
                            if st.button("Delete", key=f"del_goal_{i}"):
                                try:
                                    goals.pop(i)
                                    # Update settings with new goals
                                    user_settings.settings['goals'] = json.dumps(goals)

                                    # Save to database using the save_settings method
                                    if user_settings.save_settings(user_settings.settings):
                                        st.success("Goal added successfully!")
                                    else:
                                        st.error("Failed to save goal")
                                except Exception as e:
                                    st.error(f"Error deleting goal: {str(e)}")       

        elif st.session_state.current_page == "Investments":
            st.header("Investment Portfolio")
            
            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            # Portfolio Summary
            portfolio = investment_analyzer.get_portfolio_performance()
            if portfolio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Value", f"{currency_symbol}{portfolio['total_value']:,.2f}")
                with col2:
                    st.metric("Return", f"{portfolio['portfolio_return']*100:.1f}%")
                with col3:
                    st.metric("Risk (Std Dev)", f"{portfolio['portfolio_risk']*100:.1f}%")

            # Investment Type Selection
            investment_type = st.selectbox(
                "Investment Type",
                ["Stocks", "Company Shares", "Treasury Bills", "Bonds"]
            )

            if investment_type == "Stocks":
                st.subheader("Add Stock Investment")
                symbol = st.text_input("Stock Symbol").upper()
                if symbol:
                    try:
                        ticker = yf.Ticker(symbol)
                        current_price = ticker.history(period='1d')['Close'].iloc[-1]
                        st.write(f"Current Price: ${current_price:.2f}")

                        quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
                        if st.button("Add Stock"):
                            conn = sqlite3.connect('financial_planner.db')
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                                purchase_date, portfolio_type)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (st.session_state.username, symbol, quantity, current_price, 
                                datetime.now().strftime('%Y-%m-%d'), 'stock'))
                            conn.commit()
                            conn.close()
                            st.success(f"Added {quantity} shares of {symbol}")
                    except Exception as e:
                        st.error(f"Error fetching stock data: {str(e)}")

            elif investment_type == "Company Shares":
                st.subheader("Add Company Shares")
                company_name = st.text_input("Company Name")
                share_price = st.number_input("Share Price", min_value=0.0, step=1.0)
                quantity = st.number_input("Number of Shares", min_value=0.0, step=1.0)
                profitability = st.number_input("Annual Profitability (%)", min_value=-100.0, max_value=1000.0, step=0.1)

                if st.button("Add Company Shares"):
                    try:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                            purchase_date, portfolio_type)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (st.session_state.username, company_name, quantity, share_price, 
                            datetime.now().strftime('%Y-%m-%d'), 'company_share'))
                        conn.commit()
                        conn.close()
                        st.success(f"Added {quantity} shares of {company_name}")
                    except Exception as e:
                        st.error(f"Error adding company shares: {str(e)}")

            elif investment_type == "Treasury Bills":
                st.subheader("Add Treasury Bill Investment")
                tenure = st.selectbox("Tenure", ["91 Days", "182 Days", "364 Days"])
                amount = st.number_input("Investment Amount", min_value=0.0, step=100.0)
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
                maturity_date = st.date_input("Maturity Date")

                if st.button("Add Treasury Bill"):
                    try:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                            purchase_date, portfolio_type)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (st.session_state.username, f"T-Bill {tenure}", 1, amount, 
                            datetime.now().strftime('%Y-%m-%d'), 'treasury_bill'))
                        conn.commit()
                        conn.close()
                        st.success(f"Added Treasury Bill investment of ${amount:,.2f}")
                    except Exception as e:
                        st.error(f"Error adding treasury bill: {str(e)}")

            elif investment_type == "Bonds":
                st.subheader("Add Bond Investment")
                bond_type = st.selectbox("Bond Type", ["Government", "Corporate"])
                bond_name = st.text_input("Bond Name/ID")
                principal = st.number_input("Principal Amount", min_value=0.0, step=100.0)
                coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
                maturity_years = st.number_input("Years to Maturity", min_value=1, max_value=30, step=1)

                if st.button("Add Bond"):
                    try:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                            purchase_date, portfolio_type)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (st.session_state.username, f"{bond_type} Bond - {bond_name}", 1, 
                            principal, datetime.now().strftime('%Y-%m-%d'), 'bond'))
                        conn.commit()
                        conn.close()
                        st.success(f"Added {bond_type} Bond investment of ${principal:,.2f}")
                    except Exception as e:
                        st.error(f"Error adding bond: {str(e)}")

            # Display Current Portfolio
            st.subheader("Current Portfolio")
            conn = sqlite3.connect('financial_planner.db')
            portfolio_df = pd.read_sql_query("""
                SELECT symbol, quantity, purchase_price, purchase_date, portfolio_type
                FROM portfolios 
                WHERE username=?
                ORDER BY purchase_date DESC
            """, conn, params=(st.session_state.username,))
            conn.close()

            if not portfolio_df.empty:
                portfolio_df['Total Value'] = portfolio_df['quantity'] * portfolio_df['purchase_price']
                st.dataframe(portfolio_df, use_container_width=True)
            else:
                st.write("No investments in portfolio yet.")

            if not portfolio_df.empty:
                st.subheader("Current Portfolio")
                for index, row in portfolio_df.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    with col1:
                        st.write(f"Symbol: {row['symbol']}")
                    with col2:
                        st.write(f"Quantity: {row['quantity']}")
                    with col3:
                        st.write(f"Purchase Price: {currency_symbol}{row['purchase_price']:,.2f}")
                    with col4:
                        st.write(f"Total Value: {currency_symbol}{row['Total Value']:,.2f}")
                    with col5:
                        if st.button("Delete", key=f"del_inv_{index}"):
                            try:
                                conn = sqlite3.connect('financial_planner.db')
                                c = conn.cursor()
                                c.execute("""
                                    DELETE FROM portfolios 
                                    WHERE username=? AND symbol=? AND quantity=? AND purchase_price=?
                                """, (st.session_state.username, row['symbol'], 
                                    row['quantity'], row['purchase_price']))
                                conn.commit()
                                conn.close()
                                st.success("Investment deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting investment: {str(e)}")              

        elif st.session_state.current_page == "Debt Management":
            debt_manager.display_debt_management_ui()
            st.header("Debt Management (Bulk Payment)")
            
            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            # Add New Debt
            st.subheader("Add New Debt")
            debt_name = st.text_input("Debt Name")
            debt_amount = st.number_input("Amount", min_value=0.0, step=100.0)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)

            if st.button("Add Debt"):
                conn = sqlite3.connect('financial_planner.db')
                c = conn.cursor()
                try:
                    c.execute("""
                        INSERT INTO transactions 
                        (username, date, category, description, amount, interest_rate)
                        VALUES (?, ?, 'debt', ?, ?, ?)
                    """, (st.session_state.username, datetime.now().strftime('%Y-%m-%d'),
                        debt_name, debt_amount, interest_rate))
                    conn.commit()
                    st.success("Debt added successfully!")
                except sqlite3.Error as e:
                    st.error(f"Error adding debt: {e}")
                finally:
                    conn.close()

            # Debt Analysis
            debt_analysis = debt_manager.analyze_debt()
            if debt_analysis:
                st.subheader("Debt Overview")
                st.metric("Total Debt", f"{currency_symbol}{debt_analysis['total_debt']:,.2f}")
                if debt_analysis['average_interest'] is not None:
                    st.metric("Average Interest Rate", f"{debt_analysis['average_interest']:.1f}%")

        elif st.session_state.current_page == "Bills & Subscriptions":
            st.header("Bills & Subscriptions")
            
            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            # Add New Bill
            st.subheader("Add New Bill")
            bill_name = st.text_input("Bill Name")
            bill_amount = st.number_input("Amount", min_value=0.0, step=10.0)
            due_date = st.date_input("Due Date")
            frequency = st.selectbox("Frequency", ["Monthly", "Weekly", "Yearly"])
            reminder = st.number_input("Reminder (days before)", min_value=0, max_value=30)

            if st.button("Add Bill"):
                conn = sqlite3.connect('financial_planner.db')
                c = conn.cursor()
                c.execute("""
                    INSERT INTO bills (username, name, amount, due_date, frequency, reminder_days)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (st.session_state.username, bill_name, bill_amount,
                    due_date.strftime('%Y-%m-%d'), frequency, reminder))
                conn.commit()
                conn.close()
                st.success("Bill added successfully!")

            # Display Upcoming Bills
            st.subheader("Upcoming Bills")
            upcoming = bill_manager.get_upcoming_bills()
            if not upcoming.empty:
                # Format the amount column directly
                upcoming['amount'] = currency_symbol + upcoming['amount'].astype(str).apply(lambda x: f"{float(x):,.2f}")
                st.dataframe(upcoming[['name', 'amount', 'due_date', 'frequency']])

        elif st.session_state.current_page == "Reports & Analysis":
            st.header("Financial Reports & Analysis")
            
            # Get currency symbol at the start of the page
            currency = user_settings.settings.get('currency', 'GHS')
            currency_symbol = get_currency_symbol(currency)

            # Generate Report
            report_type = st.selectbox("Report Type", ["Monthly", "Quarterly", "Annual"])
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    report = report_generator.generate_monthly_report()

                    # Display report sections
                    if report['income_expense']:
                        st.subheader("Income vs Expenses")
                        st.write(report['income_expense'])

                    if report['budget_performance']:
                        st.subheader("Budget Performance")
                        st.write(report['budget_performance'])

                    if report['investment_performance']:
                        st.subheader("Investment Performance")
                        st.write(report['investment_performance'])

                    if report['recommendations']:
                        st.subheader("Recommendations")
                        for rec in report['recommendations']:
                            st.write(f"• {rec}")

            # In the "Reports & Analysis" section
            st.subheader("Export Data")
            export_format = st.selectbox("Export Format", ["CSV", "Excel"])
            if st.button("Export Transactions"):
                conn = sqlite3.connect('financial_planner.db')
                transactions = pd.read_sql_query("""
                    SELECT date, category, amount, description 
                    FROM transactions 
                    WHERE username=?
                """, conn, params=(st.session_state.username,))
                conn.close()

                if export_format == "CSV":
                    csv = transactions.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="transactions.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        transactions.to_excel(writer, index=False)
                    excel_data = output.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name="transactions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        elif st.session_state.current_page == "Settings":
            st.header("Settings")
            
            


            # User Preferences
            st.subheader("Preferences")
            # Get current settings first
            current_currency = user_settings.settings.get('currency', 'GHS')
            current_theme = user_settings.settings.get('theme', 'light')
            
            currency = st.selectbox("Currency", 
                          ["USD", "EUR", "GBP", "JPY", "GHS"], 
                          index=["USD", "EUR", "GBP", "JPY", "GHS"].index(current_currency))
            
            # Only perform currency conversion if the currency has changed
            if currency != current_currency:
                try:
                    # Convert all amounts in transactions
                    conn = sqlite3.connect('financial_planner.db')
                    c = conn.cursor()
                    transactions = pd.read_sql_query("""
                        SELECT id, amount FROM transactions WHERE username=?
                    """, conn, params=(st.session_state.username,))
                    
                    for _, row in transactions.iterrows():
                        new_amount = convert_amount(row['amount'], current_currency, currency)
                        c.execute("""
                            UPDATE transactions 
                            SET amount=? 
                            WHERE id=?
                        """, (new_amount, row['id']))
                    
                    conn.commit()
                    conn.close()
                    st.success(f"Converted all amounts from {current_currency} to {currency}")
                except Exception as e:
                    st.error(f"Error converting currency: {e}")
            

            # Enhanced theme selection
            theme = st.selectbox("Theme", 
                                ["light", "dark", "blue", "green", "purple", "pink"],
                                index=["light", "dark", "blue", "green", "purple", "pink"].index(
                                    current_theme if current_theme in ["light", "dark", "blue", "green", "purple", "pink"] else "dark"))
            
                        # Show theme preview
# In Settings page, add check for theme existence
            if theme in theme_colors:
                colors = theme_colors[theme]
            else:
                st.error(f"Theme '{theme}' not found, using default theme")
                colors = theme_colors['light']
                st.markdown(f"""
                    <style>
                        .theme-preview {{
                            background-color: {colors['background']};
                            color: {colors['text']};
                            padding: 1rem;
                            border-radius: 0.5rem;
                            margin: 1rem 0;
                            border: 1px solid {colors['accent']};
                        }}
                        
                        .theme-preview-sidebar {{
                            background-color: {colors['sidebar_bg']};
                            color: {colors['sidebar_text']};
                            padding: 1rem;
                            border-radius: 0.5rem;
                            margin: 0.5rem 0;
                        }}
                    </style>
                    <div class="theme-preview">
                        <h4>Main Content Preview</h4>
                        <p>This is how your main content will look.</p>
                    </div>
                    <div class="theme-preview-sidebar">
                        <h4>Sidebar Preview</h4>
                        <p>This is how your sidebar will look.</p>
                    </div>
                """, unsafe_allow_html=True)

            # Notification Settings
            st.subheader("Notifications")
            notifications = user_settings.settings.get('notifications', {})
            email_notifications = st.checkbox("Email Notifications", 
                                        value=notifications.get('email', True))
            bill_reminders = st.checkbox("Bill Reminders", 
                                    value=notifications.get('bill_reminders', True))

            if st.button("Save Settings"):
                try:
                    # Update settings dictionary
                    user_settings.settings.update({
                        'currency': currency,
                        'theme': theme,
                        'notifications': {
                            'email': email_notifications,
                            'bill_reminders': bill_reminders
                        }
                    })

                    # Apply theme immediately
                    if theme in theme_colors:
                        st.markdown(f"""
                            <style>
                                .stApp {{
                                    background-color: {theme_colors[theme]['background']};
                                    color: {theme_colors[theme]['text']};
                                }}
                            </style>
                        """, unsafe_allow_html=True)

                    # Save settings to database
                    if user_settings.save_settings(user_settings.settings):
                        st.success("Settings saved successfully! Please refresh the page for all changes to take effect.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to save settings")
                except Exception as e:
                    st.error(f"Error saving settings: {e}")


if __name__ == "__main__":
    main()
