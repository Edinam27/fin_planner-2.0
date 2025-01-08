import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import sqlite3
import json
import numpy as np
from decimal import Decimal
import yfinance as yf  # For stock data
import requests
import calendar
from forex_python.converter import CurrencyRates
import pycountry
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shutil
from datetime import datetime, timedelta
from io import BytesIO  # Add this import if not already present
import xlsxwriter
import os
import time


# Database setup enhancements
def init_db():
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    
    # Enhanced users table
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
    
    # Enhanced transactions table

    # In the transactions table creation:
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  date TEXT,
                  category TEXT,
                  subcategory TEXT,
                  amount REAL,
                  description TEXT,
                  payment_method TEXT,
                  recurring BOOLEAN,
                  tags TEXT,
                  transaction_type TEXT,
                  attachment_path TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    
    # Investment portfolio table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  symbol TEXT,
                  quantity REAL,
                  purchase_price REAL,
                  purchase_date TEXT,
                  portfolio_type TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    
    # Bills and subscriptions table
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
    
    # Financial reports table
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  report_type TEXT,
                  report_date TEXT,
                  report_data TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    
    # Add system_logs table
    c.execute('''CREATE TABLE IF NOT EXISTS system_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  level TEXT,
                  message TEXT,
                  source TEXT,
                  details TEXT)''')
        
    conn.commit()
    conn.close()

# Enhanced User Settings and Preferences
class UserSettings:
    def __init__(self, username):
        self.username = username
        self.settings = self._load_settings()

    def _load_settings(self):
        try:
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            c.execute("SELECT settings FROM users WHERE username=?", (self.username,))
            result = c.fetchone()
            conn.close()
            
            if result and result[0]:
                settings = json.loads(result[0])
                # Ensure goals is stored as JSON string
                if 'goals' in settings and not isinstance(settings['goals'], str):
                    settings['goals'] = json.dumps(settings['goals'])
                return settings
            return self._default_settings()
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
            return self._default_settings()
        
    def save_settings(self, settings):
        try:
            # Ensure goals is JSON string before saving
            if 'goals' in settings and not isinstance(settings['goals'], str):
                settings['goals'] = json.dumps(settings['goals'])
                
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            c.execute("""
                UPDATE users 
                SET settings=? 
                WHERE username=?
            """, (json.dumps(settings), self))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")
            return False
    
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
                'threshold': 80,  # Alert when category reaches 80% of budget
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
        


# Enhanced Financial Analysis Tools
class FinancialAnalyzer:
    def __init__(self, username):
        self.username = username
        
    def calculate_net_worth(self):
        """Calculate total net worth including assets and liabilities"""
        conn = sqlite3.connect('financial_planner.db')
        c = conn.cursor()
        
        # Get assets
        c.execute("""
            SELECT SUM(amount) FROM transactions 
            WHERE username=? AND category='income'
        """, (self.username,))
        assets = c.fetchone()[0] or 0
        
        # Get liabilities
        c.execute("""
            SELECT SUM(amount) FROM transactions 
            WHERE username=? AND category='debt'
        """, (self.username,))
        liabilities = c.fetchone()[0] or 0
        
        conn.close()
        return assets - liabilities
    
    def analyze_spending_patterns(self):
        """Analyze spending patterns and provide insights."""
        conn = sqlite3.connect('financial_planner.db')
        current_month = datetime.now().strftime('%Y-%m')

        # Get total income directly from transaction_type
        income_df = pd.read_sql_query("""
            SELECT SUM(amount) as income
            FROM transactions 
            WHERE username=? 
            AND transaction_type='Income'
            AND strftime('%Y-%m', date)=?
        """, conn, params=(self.username, current_month))

        # Get total expenses directly from transaction_type
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

        # Calculate monthly savings correctly
        monthly_savings = total_income - total_expenses

        return {
            'monthly_savings': monthly_savings,
            'total_income': total_income,
            'total_expenses': total_expenses
        }


        
# Enhanced Investment Analysis
class InvestmentAnalyzer:
    def __init__(self, username):
        self.username = username
        
    def get_portfolio_performance(self):
        """Calculate portfolio performance including returns and risk metrics"""
        conn = sqlite3.connect('financial_planner.db')
        portfolio = pd.read_sql_query("""
            SELECT * FROM portfolios WHERE username=?
        """, conn, params=(self.username,))
        conn.close()
        
        if portfolio.empty:
            return None
        
        total_value = 0
        returns = []
        
        for _, position in portfolio.iterrows():
            try:
                # Get current price using yfinance
                ticker = yf.Ticker(position['symbol'])
                history = ticker.history(period='1d')
                
                # Check if we got any data
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                else:
                    # Use purchase price if no current data available
                    current_price = position['purchase_price']
                    st.warning(f"Could not fetch current price for {position['symbol']}, using purchase price")
                
                # Calculate position value and return
                position_value = current_price * position['quantity']
                position_return = (current_price - position['purchase_price']) / position['purchase_price']
                
                total_value += position_value
                returns.append(position_return)
            except Exception as e:
                st.warning(f"Error processing {position['symbol']}: {str(e)}")
                # Use purchase price as fallback
                total_value += position['purchase_price'] * position['quantity']
                returns.append(0.0)
        
        if not returns:
            return {
                'total_value': total_value,
                'portfolio_return': 0.0,
                'portfolio_risk': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Calculate portfolio metrics
        portfolio_return = np.mean(returns)
        portfolio_risk = np.std(returns)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'total_value': total_value,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_investment_recommendations(self, risk_profile):
        """Generate investment recommendations based on risk profile"""
        risk_profiles = {
            'conservative': {
                'bonds': 60,
                'stocks': 30,
                'cash': 10,
                'recommendations': [
                    'Treasury bonds',
                    'High-grade corporate bonds',
                    'Blue-chip dividend stocks'
                ]
            },
            'moderate': {
                'bonds': 40,
                'stocks': 50,
                'cash': 10,
                'recommendations': [
                    'Index funds',
                    'Balanced mutual funds',
                    'Mixed corporate bonds'
                ]
            },
            'aggressive': {
                'bonds': 20,
                'stocks': 70,
                'cash': 10,
                'recommendations': [
                    'Growth stocks',
                    'Small-cap funds',
                    'International equity'
                ]
            }
        }
        return risk_profiles.get(risk_profile.lower())

# Enhanced Debt Management
class DebtManager:
    def __init__(self, username):
        self.username = username
    
    def analyze_debt(self):
        """Analyze debt and provide repayment strategies"""
        conn = sqlite3.connect('financial_planner.db')
        debts = pd.read_sql_query("""
            SELECT * FROM transactions 
            WHERE username=? AND category='debt'
        """, conn, params=(self.username,))
        conn.close()
        
        if debts.empty:
            return None
        
        # Calculate key metrics
        total_debt = debts['amount'].sum()
        avg_interest = debts['interest_rate'].mean() if 'interest_rate' in debts else 0
        
        # Generate repayment strategies
        strategies = {
            'avalanche': self._avalanche_strategy(debts),
            'snowball': self._snowball_strategy(debts),
            'consolidation': self._consolidation_strategy(debts)
        }
        
        return {
            'total_debt': total_debt,
            'average_interest': avg_interest,
            'strategies': strategies
        }
    
    def _avalanche_strategy(self, debts):
        """Calculate highest interest first strategy"""
        # Implementation details...
        pass
    
    def _snowball_strategy(self, debts):
        """Calculate smallest balance first strategy"""
        # Implementation details...
        pass
    
    def _consolidation_strategy(self, debts):
        """Calculate debt consolidation strategy"""
        # Implementation details...
        pass

# Enhanced Bill Management
class BillManager:
    def __init__(self, username):
        self.username = username
    
    def get_upcoming_bills(self, days=30):
        """Get list of upcoming bills"""
        conn = sqlite3.connect('financial_planner.db')
        current_date = datetime.now().date()
        end_date = current_date + timedelta(days=days)
        
        bills = pd.read_sql_query("""
            SELECT * FROM bills 
            WHERE username=? AND due_date BETWEEN ? AND ?
        """, conn, params=(self.username, current_date, end_date))
        conn.close()
        
        return bills.sort_values('due_date')
    
    def set_bill_reminder(self, bill_id, reminder_days):
        """Set reminder for bill payment"""
        conn = sqlite3.connect('financial_planner.db')
        c = conn.cursor()
        c.execute("""
            UPDATE bills 
            SET reminder_days=? 
            WHERE id=? AND username=?
        """, (reminder_days, bill_id, self.username))
        conn.commit()
        conn.close()

# Enhanced Report Generator
class ReportGenerator:
    def __init__(self, username):
        self.username = username
    
    def generate_monthly_report(self, month=None):
        """Generate comprehensive monthly financial report"""
        if month is None:
            month = datetime.now().strftime('%Y-%m')
        
        # Get all relevant data
        analyzer = FinancialAnalyzer(self.username)
        investment_analyzer = InvestmentAnalyzer(self.username)
        
        # Generate report sections
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
        
        # Save report
        self._save_report(report, 'monthly')
        return report
    
    def _analyze_income_expenses(self, month):
        """Analyze income and expenses for the month"""
        # Implementation details...
        pass
    
    def _analyze_budget_performance(self, month):
        """Analyze budget performance for the month"""
        # Implementation details...
        pass
    
    def _analyze_savings_progress(self):
        """Analyze progress towards savings goals"""
        # Implementation details...
        pass
    
    def _generate_recommendations(self):
        """Generate personalized financial recommendations"""
        # Implementation details...
        pass
    
    def _save_report(self, report_data, report_type):
        """Save report to database"""
        conn = sqlite3.connect('financial_planner.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO reports (username, report_type, report_date, report_data)
            VALUES (?, ?, ?, ?)
        """, (self.username, report_type, datetime.now().strftime('%Y-%m-%d'),
              json.dumps(report_data)))
        conn.commit()
        conn.close()
        
        
class AdminDashboard:
    def __init__(self):
        self.conn = sqlite3.connect('financial_planner.db')
        self.init_session_state()
        self.financial_analyzer = FinancialAnalyzer('admin')
        self.investment_analyzer = InvestmentAnalyzer('admin')
        self.report_generator = ReportGenerator('admin')
        self.debt_manager = DebtManager('admin')
        
        # Initialize system logs table if it doesn't exist
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS system_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT,
                    source TEXT,
                    details TEXT)''')
        self.conn.commit()

    def init_session_state(self):
        if "admin_authenticated" not in st.session_state:
            st.session_state.admin_authenticated = False
        if "admin_username" not in st.session_state:
            st.session_state.admin_username = None
        if "admin_current_page" not in st.session_state:
            st.session_state.admin_current_page = "Dashboard"
        if "admin_settings" not in st.session_state:
            st.session_state.admin_settings = {
                'theme': 'light',
                'show_advanced_metrics': True,
                'refresh_interval': 300  # 5 minutes
            }

    def admin_login(self):
        st.title("Admin Dashboard")
        
        with st.form("admin_login"):
            username = st.text_input("Admin Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username == "admin" and hashlib.sha256(password.encode()).hexdigest() == hashlib.sha256("prettyFLACO1$".encode()).hexdigest():
                    st.session_state.admin_authenticated = True
                    st.session_state.admin_username = username
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")

    def user_management(self):
        st.subheader("User Management")
        
        # Display user statistics
        users_df = pd.read_sql_query("SELECT * FROM users", self.conn)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", len(users_df))
        with col2:
            active_users = len(users_df[users_df['last_login'] > (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')])
            st.metric("Active Users (30 days)", active_users)
        with col3:
            new_users = len(users_df[users_df['created_date'] > (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')])
            st.metric("New Users (30 days)", new_users)

        # User actions
        if st.checkbox("Show User Details"):
            st.dataframe(users_df[['username', 'email', 'created_date', 'last_login']])
            
            # User deletion
            user_to_delete = st.selectbox("Select user to delete", users_df['username'])
            if st.button("Delete User"):
                if st.checkbox("Confirm deletion"):
                    self.delete_user(user_to_delete)
                    st.success(f"User {user_to_delete} deleted successfully")
                    st.rerun()

    def transaction_monitoring(self):
        st.subheader("Transaction Monitoring")
        
        transactions_df = pd.read_sql_query("""
            SELECT username, date, category, amount, transaction_type 
            FROM transactions 
            ORDER BY date DESC
        """, self.conn)
        
        # Transaction statistics
        total_volume = transactions_df['amount'].sum()
        st.metric("Total Transaction Volume", f"${total_volume:,.2f}")
        
        # Transaction filters
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("Date Range", 
                                     [datetime.now() - timedelta(days=30), datetime.now()])
        with col2:
            transaction_type = st.multiselect("Transaction Type", 
                                            ["Income", "Expense"], 
                                            default=["Income", "Expense"])
        
        # Filtered transactions
        filtered_df = transactions_df[
            (transactions_df['date'].between(date_range[0].strftime('%Y-%m-%d'), 
                                           date_range[1].strftime('%Y-%m-%d'))) &
            (transactions_df['transaction_type'].isin(transaction_type))
        ]
        
        st.dataframe(filtered_df)
        
        # Transaction visualization
        fig = px.line(filtered_df, x='date', y='amount', color='transaction_type',
                     title='Transaction Trends')
        st.plotly_chart(fig)

    def system_maintenance(self):
        st.subheader("System Maintenance")
        
        # Database backup
        if st.button("Backup Database"):
            if self.backup_database():
                st.success("Database backed up successfully")
            else:
                st.error("Backup failed")
        
        # Database statistics
        db_stats = self.get_database_stats()
        st.json(db_stats)
        
        # System logs (if implemented)
        st.subheader("System Logs")
        # Implement system logging functionality

    def delete_user(self, username):
        c = self.conn.cursor()
        try:
            # Delete user's data from all tables
            tables = ['users', 'transactions', 'portfolios', 'bills', 'reports']
            for table in tables:
                c.execute(f"DELETE FROM {table} WHERE username=?", (username,))
            self.conn.commit()
            
            # Log the event
            self.log_system_event(
                'INFO',
                f'User {username} deleted',
                'user_management',
                {'affected_tables': tables}
            )
            return True
        except Exception as e:
            self.log_system_event(
                'ERROR',
                f'Failed to delete user {username}',
                'user_management',
                {'error': str(e)}
            )
            st.error(f"Error deleting user: {e}")
            self.conn.rollback()
            return False

    def backup_database(self):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f'admin_backup_{timestamp}.db'
            shutil.copy2('financial_planner.db', backup_file)
            return True
        except Exception as e:
            st.error(f"Backup error: {e}")
            return False

    def get_database_stats(self):
        c = self.conn.cursor()
        stats = {}
        
        # Get table sizes
        tables = ['users', 'transactions', 'portfolios', 'bills', 'reports']
        for table in tables:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = c.fetchone()[0]
        
        # Get database file size
        import os
        stats["database_size_mb"] = os.path.getsize('financial_planner.db') / (1024 * 1024)
        
        return stats

    def run(self):
        if not st.session_state.admin_authenticated:
            self.admin_login()
        else:
            st.sidebar.title("Admin Dashboard")
            
            # Enhanced navigation with icons and organization
            pages = {
                "Dashboard": {"icon": "📊", "func": self.admin_dashboard},
                "User Management": {"icon": "👥", "func": self.user_management},
                "Transaction Monitoring": {"icon": "💳", "func": self.transaction_monitoring},
                "Investment Overview": {"icon": "📈", "func": self.investment_overview},
                "Debt Analytics": {"icon": "💰", "func": self.debt_analytics},
                "System Maintenance": {"icon": "🔧", "func": self.system_maintenance},
                "Reports & Analytics": {"icon": "📑", "func": self.reports_analytics},
                "Settings": {"icon": "⚙️", "func": self.admin_settings}
            }
            
            selection = st.sidebar.selectbox(
                "Navigation",
                pages.keys(),
                format_func=lambda x: f"{pages[x]['icon']} {x}"
            )
            
            # Admin info and logout
            st.sidebar.divider()
            st.sidebar.info(f"Logged in as: {st.session_state.admin_username}")
            if st.sidebar.button("🚪 Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()
            
            # Execute selected page function
            pages[selection]["func"]()
                
                
    def admin_dashboard(self):
        st.title("Admin Dashboard Overview")
        
        # Key Metrics Section
        st.subheader("Key Metrics")
        metrics = self.get_key_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Users", 
                metrics['users_count'], 
                delta=str(metrics['new_users_24h']) if metrics['new_users_24h'] > 0 else None
            )
        with col2:
            st.metric("Active Users (24h)", metrics['active_users_24h'])
        with col3:
            st.metric(
                "Total Transactions", 
                metrics['transactions_count'],
                delta=str(metrics['transactions_24h']) if metrics['transactions_24h'] > 0 else None
            )
        with col4:
            st.metric("System Health", metrics['system_health_score'])

        # Activity Monitoring
        st.subheader("Real-time Activity Monitor")
        col1, col2 = st.columns(2)
        
        with col1:
            # User Activity Chart
            user_activity = self.get_user_activity_data()
            fig = px.line(user_activity, x='timestamp', y='active_users',
                         title='User Activity (Last 24 Hours)')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Transaction Volume Chart
            transaction_volume = self.get_transaction_volume_data()
            fig = px.bar(transaction_volume, x='hour', y='volume',
                        title='Transaction Volume by Hour')
            st.plotly_chart(fig, use_container_width=True)

        # System Health Dashboard
        self.display_system_health()
        
    def get_user_activity_data(self):
        """Get user activity data for the last 24 hours"""
        try:
            # Get timestamp from 24 hours ago
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Query user activity by hour
            activity_df = pd.read_sql_query("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', last_login) as timestamp,
                    COUNT(DISTINCT username) as active_users
                FROM users 
                WHERE last_login >= ?
                GROUP BY strftime('%Y-%m-%d %H:00:00', last_login)
                ORDER BY timestamp
            """, self.conn, params=(yesterday,))
            
            # If no data, create empty DataFrame with zero values
            if activity_df.empty:
                hours = pd.date_range(
                    start=yesterday,
                    end=datetime.now(),
                    freq='H'
                )
                activity_df = pd.DataFrame({
                    'timestamp': hours,
                    'active_users': [0] * len(hours)
                })
            
            return activity_df
        except Exception as e:
            st.error(f"Error fetching user activity data: {str(e)}")
            return pd.DataFrame({'timestamp': [], 'active_users': []})

    def get_transaction_volume_data(self):
        """Get transaction volume data by hour for the last 24 hours"""
        try:
            # Get timestamp from 24 hours ago
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Query transaction volume by hour
            volume_df = pd.read_sql_query("""
                SELECT 
                    strftime('%H', date) as hour,
                    COUNT(*) as volume
                FROM transactions 
                WHERE date >= ?
                GROUP BY strftime('%H', date)
                ORDER BY hour
            """, self.conn, params=(yesterday,))
            
            # If no data, create empty DataFrame with zero values
            if volume_df.empty:
                volume_df = pd.DataFrame({
                    'hour': [str(h).zfill(2) for h in range(24)],
                    'volume': [0] * 24
                })
            
            return volume_df
        except Exception as e:
            st.error(f"Error fetching transaction volume data: {str(e)}")
            return pd.DataFrame({'hour': [], 'volume': []})
        
    def get_key_metrics(self):
        """Get key system metrics"""
        try:
            current_time = datetime.now()
            yesterday = current_time - timedelta(days=1)
            
            # User metrics
            users_df = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_users,
                    SUM(CASE WHEN created_date >= ? THEN 1 ELSE 0 END) as new_users_24h,
                    SUM(CASE WHEN last_login >= ? THEN 1 ELSE 0 END) as active_users_24h
                FROM users
            """, self.conn, params=(yesterday.strftime('%Y-%m-%d %H:%M:%S'),
                                yesterday.strftime('%Y-%m-%d %H:%M:%S')))
            
            # Transaction metrics
            transactions_df = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN date >= ? THEN 1 ELSE 0 END) as transactions_24h
                FROM transactions
            """, self.conn, params=(yesterday.strftime('%Y-%m-%d %H:%M:%S'),))
            
            # Calculate system health score
            system_health_score = self.calculate_system_health_score()
            
            # Convert numpy.int64 to native Python types
            return {
                'users_count': int(users_df['total_users'].iloc[0]),
                'new_users_24h': int(users_df['new_users_24h'].iloc[0]),
                'active_users_24h': int(users_df['active_users_24h'].iloc[0]),
                'transactions_count': int(transactions_df['total_transactions'].iloc[0]),
                'transactions_24h': int(transactions_df['transactions_24h'].iloc[0]),
                'system_health_score': f"{system_health_score}%"
            }
        except Exception as e:
            st.error(f"Error fetching metrics: {str(e)}")
            return {
                'users_count': 0,
                'new_users_24h': 0,
                'active_users_24h': 0,
                'transactions_count': 0,
                'transactions_24h': 0,
                'system_health_score': "N/A"
            }
        
        
    def calculate_system_health_score(self):
        """Calculate overall system health score"""
        try:
            # Example scoring system
            scores = []
            
            # Database size score
            db_size = os.path.getsize('financial_planner.db') / (1024 * 1024)  # MB
            scores.append(100 if db_size < 100 else (90 if db_size < 500 else 70))
            
            # Query performance score
            start_time = time.time()
            pd.read_sql_query("SELECT COUNT(*) FROM transactions", self.conn)
            query_time = time.time() - start_time
            scores.append(100 if query_time < 0.1 else (90 if query_time < 0.5 else 70))
            
            # Return average score
            return int(sum(scores) / len(scores))
        except Exception:
            return 0

    def display_system_health(self):
        st.subheader("System Health Monitor")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Database metrics
            db_size = os.path.getsize('financial_planner.db') / (1024 * 1024)
            st.metric("Database Size", f"{db_size:.2f} MB")
            if db_size > 1000:
                st.warning("Database size is getting large")
                
        with col2:
            # Performance metrics
            start_time = time.time()
            pd.read_sql_query("SELECT COUNT(*) FROM transactions", self.conn)
            query_time = time.time() - start_time
            st.metric("Query Response Time", f"{query_time:.3f}s")
            
        with col3:
            try:
                # Check for recent errors
                error_count = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM system_logs WHERE level='ERROR' AND timestamp >= datetime('now', '-24 hours')",
                    self.conn
                ).iloc[0]['count']
                st.metric("24h Errors", error_count)
                if error_count > 0:
                    st.error(f"Found {error_count} errors")
            except Exception as e:
                st.metric("24h Errors", "N/A")
                if st.session_state.admin_settings.get('show_advanced_metrics', True):
                    st.info("System logs monitoring not initialized")
                    
    def log_system_event(self, level, message, source=None, details=None):
        """Log system events to the database"""
        try:
            c = self.conn.cursor()
            c.execute("""
                INSERT INTO system_logs (timestamp, level, message, source, details)
                VALUES (datetime('now'), ?, ?, ?, ?)
            """, (level, message, source, json.dumps(details) if details else None))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error logging system event: {str(e)}")
            return False

    def investment_overview(self):
        st.title("Investment Analytics")
        
        try:
            # Get all investment data with unique column names
            investments_df = pd.read_sql_query("""
                SELECT 
                    p.id,
                    p.username,
                    p.symbol,
                    p.quantity,
                    p.purchase_price,
                    p.purchase_date,
                    p.portfolio_type,
                    u.email,
                    u.life_stage,
                    u.risk_profile
                FROM portfolios p
                LEFT JOIN users u ON p.username = u.username
                ORDER BY p.purchase_date DESC
            """, self.conn)
            
            if not investments_df.empty:
                # Portfolio statistics
                total_investments = (investments_df['quantity'] * investments_df['purchase_price']).sum()
                avg_investment = total_investments / len(investments_df) if len(investments_df) > 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Investment Value", f"${total_investments:,.2f}")
                with col2:
                    st.metric("Total Portfolios", len(investments_df))
                with col3:
                    st.metric("Average Investment", f"${avg_investment:,.2f}")
                
                # Investment distribution chart
                st.subheader("Investment Distribution")
                fig = px.pie(investments_df, 
                            values='quantity', 
                            names='portfolio_type',
                            title='Investment Distribution by Type')
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk profile distribution
                st.subheader("Risk Profile Distribution")
                risk_dist = investments_df.groupby('risk_profile').size().reset_index(name='count')
                fig = px.bar(risk_dist,
                            x='risk_profile',
                            y='count',
                            title='Investment Distribution by Risk Profile')
                st.plotly_chart(fig, use_container_width=True)
                
                # User investment table
                st.subheader("Investment Details")
                display_cols = ['username', 'symbol', 'quantity', 'purchase_price', 
                            'purchase_date', 'portfolio_type', 'risk_profile']
                st.dataframe(investments_df[display_cols], use_container_width=True)
                
            else:
                st.info("No investment data available")
                
        except Exception as e:
            st.error(f"Error loading investment data: {str(e)}")
            self.log_system_event(
                'ERROR',
                'Failed to load investment overview',
                'investment_overview',
                {'error': str(e)}
            )
        
        
    def debt_analytics(self):
        st.title("Debt Analytics")
        
        # Get all debt data
        debts_df = pd.read_sql_query("""
            SELECT t.*, u.username 
            FROM transactions t
            JOIN users u ON t.username = u.username
            WHERE t.category = 'debt'
        """, self.conn)
        
        # Debt statistics
        total_debt = debts_df['amount'].sum()
        st.metric("Total System Debt", f"${total_debt:,.2f}")
        
        # Debt distribution chart
        fig = px.bar(debts_df, 
                    x='username', 
                    y='amount',
                    title='Debt Distribution by User')
        st.plotly_chart(fig)

    def reports_analytics(self):
        st.title("System Reports & Analytics")
        
        report_type = st.selectbox(
            "Select Report Type",
            ["User Growth", "Transaction Volume", "Investment Performance", "System Usage"]
        )
        
        if report_type == "User Growth":
            self.generate_user_growth_report()
        elif report_type == "Transaction Volume":
            self.generate_transaction_volume_report()
        elif report_type == "Investment Performance":
            self.generate_investment_performance_report()
        elif report_type == "System Usage":
            self.generate_system_usage_report()
            
    def display_system_health(self):
        st.subheader("System Health")
        
        # Database size and performance
        db_size = os.path.getsize('financial_planner.db') / (1024 * 1024)  # MB
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Database Size", f"{db_size:.2f} MB")
            if db_size > 1000:  # 1GB warning
                st.warning("Database size is getting large, consider optimization")
                
        with col2:
            # Check for recent errors
            error_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM system_logs WHERE level='ERROR' AND timestamp >= datetime('now', '-24 hours')",
                self.conn
            ).iloc[0]['count']
            st.metric("24h Errors", error_count)
            if error_count > 0:
                st.error(f"Found {error_count} errors in the last 24 hours")
                
    def admin_settings(self):
        st.title("Admin Settings")
        
        with st.form("admin_settings"):
            theme = st.selectbox(
                "Dashboard Theme",
                ["light", "dark"],
                index=0 if st.session_state.admin_settings['theme'] == 'light' else 1
            )
            
            show_advanced = st.checkbox(
                "Show Advanced Metrics",
                value=st.session_state.admin_settings['show_advanced_metrics']
            )
            
            refresh_interval = st.slider(
                "Dashboard Refresh Interval (seconds)",
                min_value=60,
                max_value=3600,
                value=st.session_state.admin_settings['refresh_interval'],
                step=60
            )
            
            if st.form_submit_button("Save Settings"):
                st.session_state.admin_settings.update({
                    'theme': theme,
                    'show_advanced_metrics': show_advanced,
                    'refresh_interval': refresh_interval
                })
                st.success("Settings saved successfully!")
                time.sleep(1)
                st.rerun()
        
        
        
# Add this to handle password reset
def reset_password(token, new_password):
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    
    # Check if token is valid and not expired
    c.execute("""
        SELECT username FROM users 
        WHERE reset_token=? AND reset_token_expiry > ?
    """, (token, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    result = c.fetchone()
    
    if result:
        username = result[0]
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        
        # Update password and clear reset token
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


# Add password complexity requirements
def validate_password(password):
    if len(password) < 8:
        return False
    if not re.search("[a-z]", password):
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    return True

# Add session timeout
def check_session_timeout():
    if 'last_activity' in st.session_state:
        if (datetime.now() - st.session_state.last_activity).total_seconds() > 1800:  # 30 minutes
            st.session_state.authenticated = False
            return True
    st.session_state.last_activity = datetime.now()
    return False
    
# Add interactive charts
def create_spending_trend_chart(data):
    fig = px.line(data, x='date', y='amount', 
                  color='category',
                  title='Spending Trends Over Time')
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
        # Implement PDF export
        pass
    
# In the "Budget Tracking" section
def check_budget_alerts(username):
    conn = sqlite3.connect('financial_planner.db')
    current_month = datetime.now().strftime('%Y-%m')
    
    # Get spending by category
    spending = pd.read_sql_query("""
        SELECT category, SUM(amount) as spent
        FROM transactions 
        WHERE username=? AND strftime('%Y-%m', date)=?
        GROUP BY category
    """, conn, params=(username, current_month))
    conn.close()
    
    # Get budget settings
    user_settings = UserSettings(username)
    budget = user_settings.settings.get('budget', {})
    
    alerts = []
    for _, row in spending.iterrows():
        if row['category'] in budget:
            budget_amount = budget[row['category']]
            if row['spent'] > budget_amount * 0.8:  # 80% threshold
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
    
    # Add custom categories
    c.execute('''CREATE TABLE IF NOT EXISTS custom_categories
                 (username TEXT,
                  category_name TEXT,
                  category_type TEXT,
                  PRIMARY KEY (username, category_name))''')
    conn.commit()
    conn.close()

def display_welcome_page():
    st.title("Welcome to Financial Planner")
    st.subheader("Achieve Your Financial Goals with Ease")

    st.write("""
    This app is designed to provide you with a complete overview of your financial life.
    From tracking your income and expenses to managing debts, investments, and savings goals,
    we've got you covered. Here's how you can make the most of it:
    """)

    st.markdown("""
    ### Features:
    - **Dashboard Overview**: Get a summary of your net worth, income vs expenses, and upcoming bills.
    - **Transactions**: Add, edit, and manage your daily transactions.
    - **Budget Tracking**: Stay within your planned expenses.
    - **Investments**: Track the performance of your portfolio.
    - **Debt Management**: Keep tabs on your liabilities.
    - **Goals**: Set and track financial milestones.
    - **Reports**: Generate insights into your financial health.
    """)

    st.info("Tip: Use the sidebar to navigate through the app sections. Each page provides a detailed explanation to guide you.")

def explain_dashboard_charts(charts):
    st.subheader("Dashboard Overview")

    # Net Worth Chart
    if 'net_worth' in charts:
        st.plotly_chart(charts['net_worth'], use_container_width=True)
        st.markdown("""
        **Net Worth Breakdown**: 
        This chart shows your total assets and liabilities, and calculates your net worth.
        """)
    else:
        st.info("No net worth data available.")

    # Spending Distribution Chart
    if 'spending' in charts:
        st.plotly_chart(charts['spending'], use_container_width=True)
        st.markdown("""
        **Spending Distribution**: 
        Understand how you allocate your expenses across categories.
        """)
    else:
        st.info("No spending data available for the last 30 days.")

    # Income vs Expenses Trend
    if 'trend' in charts:
        st.plotly_chart(charts['trend'], use_container_width=True)
        st.markdown("""
        **Income vs Expenses Trend**: 
        This chart compares your monthly income to your expenses, helping you track savings or overspending trends.
        """)
    else:
        st.info("No trend data available.")
        

def get_debt_details(self):
    try:
        debts = pd.read_sql_query("""
            SELECT 
                id,
                description as name,  # Using description instead of name
                amount,
                interest_rate,
                minimum_payment,
                date as start_date 
            FROM transactions 
            WHERE username=? AND category='debt' AND amount > 0 
            ORDER BY interest_rate DESC
        """, self.conn, params=(self.username,))
        return debts
    except Exception as e:
        st.error(f"Error retrieving debt details: {str(e)}")
        return pd.DataFrame()




if __name__ == "__main__":
    admin = AdminDashboard()
    admin.run()
    ##main()

            