import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quant Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .css-1d391kg { background-color: #1e293b; }
    div[data-testid="stMetricValue"] { color: #f1f5f9; }
    </style>
""", unsafe_allow_html=True)

STRATEGY_COLORS = [
    '#3b82f6', '#10b981', '#f97316', '#ef4444', '#8b5cf6', 
    '#06b6d4', '#eab308', '#ec4899', '#14b8a6', '#f43f5e'
]

# ---------------------------------------------------------
# 2. Data Processing Logic
# ---------------------------------------------------------
def process_tradingview_csv(uploaded_file, color):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        
        df_exits = df[df['ประเภท'].astype(str).str.contains('ออกจากสถานะ')].copy()
        if len(df_exits) == 0: return None
            
        df_exits['Datetime'] = pd.to_datetime(df_exits['วันที่ และ เวลา'])
        df_exits['Date'] = df_exits['Datetime'].dt.date
        
        df_exits['PnL'] = pd.to_numeric(df_exits['P&L สุทธิ THB'].astype(str).str.replace(',', ''), errors='coerce')
        df_exits = df_exits.dropna(subset=['PnL']).sort_values('Datetime').reset_index(drop=True)
        
        df_exits['Trade_Index'] = df_exits.index + 1
        df_exits['Running_PnL'] = df_exits['PnL'].cumsum()
        df_exits['Peak'] = df_exits['Running_PnL'].cummax()
        df_exits['Drawdown'] = df_exits['Running_PnL'] - df_exits['Peak']
        
        total_trades = len(df_exits)
        winning_trades = df_exits[df_exits['PnL'] > 0]
        losing_trades = df_exits[df_exits['PnL'] <= 0]
        
        gross_profit = winning_trades['PnL'].sum()
        gross_loss = abs(losing_trades['PnL'].sum())
        net_profit = df_exits['Running_PnL'].iloc[-1]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        max_dd = df_exits['Drawdown'].min()
        recovery_factor = net_profit / abs(max_dd) if max_dd < 0 else float('inf')
        avg_trade = net_profit / total_trades if total_trades > 0 else 0
        
        df_exits['Is_Loss'] = (df_exits['PnL'] <= 0).astype(int)
        df_exits['Consecutive_Loss_Block'] = (df_exits['Is_Loss'] != df_exits['Is_Loss'].shift()).cumsum()
        max_cons_losses = df_exits[df_exits['Is_Loss'] == 1].groupby('Consecutive_Loss_Block').size().max()
        if pd.isna(max_cons_losses): max_cons_losses = 0

        daily_pnl = df_exits.groupby('Date')['PnL'].sum()
        mean_daily = daily_pnl.mean()
        std_daily = daily_pnl.std()
        sharpe_ratio = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
        
        # --- NEW: Drawdown Duration Analysis ---
        df_exits['Is_DD'] = df_exits['Drawdown'] < 0
        df_exits['Peak_ID'] = (df_exits['Drawdown'] == 0).cumsum()
        
        dd_periods = []
        for peak_id, group in df_exits[df_exits['Is_DD']].groupby('Peak_ID'):
            peak_rows = df_exits[(df_exits['Peak_ID'] == peak_id) & (~df_exits['Is_DD'])]
            start_date = peak_rows['Date'].iloc[-1] if not peak_rows.empty else group['Date'].min()
            
            last_group_idx = group.index[-1]
            recovered = False
            if last_group_idx + 1 < len(df_exits):
                end_date = df_exits.loc[last_group_idx + 1, 'Date']
                recovered = True
            else:
                end_date = group['Date'].max() # Ongoing Drawdown
                
            depth = group['Drawdown'].min()
            duration = (end_date - start_date).days
            
            dd_periods.append({
                'Start Date': start_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d') if recovered else f"Ongoing",
                'Duration (Days)': duration,
                'Max Depth (THB)': depth,
                'Status': 'Recovered' if recovered else 'Not Recovered'
            })
            
        # Get Top 3 Longest Drawdowns
        top_drawdowns = sorted(dd_periods, key=lambda x: x['Duration (Days)'], reverse=True)[:3]
        
        return {
            'id': uploaded_file.name,
            'name': uploaded_file.name.replace('.csv', ''),
            'color': color,
            'data': df_exits,
            'stats': {
                'Net Profit': net_profit,
                'Total Trades': total_trades,
                'Win Rate (%)': win_rate,
                'Profit Factor': profit_factor,
                'Avg Trade': avg_trade,
                'Max Drawdown': max_dd,
                'Sharpe Ratio': sharpe_ratio,
                'Recovery Factor': recovery_factor,
                'Max Cons. Loss': int(max_cons_losses)
            },
            'top_drawdowns': top_drawdowns
        }
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

# ---------------------------------------------------------
# 3. Monte Carlo Simulation Engine
# ---------------------------------------------------------
def run_monte_carlo(df_exits, num_simulations=1000):
    trade_returns = df_exits['PnL'].values
    num_trades = len(trade_returns)
    np.random.seed(42)
    simulations = np.random.choice(trade_returns, size=(num_simulations, num_trades), replace=True)
    cumulative_pnl = np.cumsum(simulations, axis=1)
    
    fig = go.Figure()
    for i in range(min(100, num_simulations)):
        fig.add_trace(go.Scatter(y=cumulative_pnl[i, :], mode='lines', line=dict(color='gray', width=1), opacity=0.1, showlegend=False))
        
    mean_path = np.mean(cumulative_pnl, axis=0)
    fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='blue', width=2, dash='dash'), name='Average Path'))
    
    original_cumulative = np.cumsum(trade_returns)
    fig.add_trace(go.Scatter(y=original_cumulative, mode='lines', line=dict(color='red', width=2), name='Original Backtest'))
    
    fig.update_layout(title=f"Monte Carlo Simulation ({num_simulations} Runs)", xaxis_title="Trade Number", yaxis_title="Cumulative P&L", template="plotly_dark", height=400, showlegend=True, margin=dict(l=0, r=0, t=40, b=0))
    
    ending_pnls = cumulative_pnl[:, -1]
    win_prob = np.mean(ending_pnls > 0) * 100
    median_pnl = np.median(ending_pnls)
    worst_case_pnl = np.percentile(ending_pnls, 5)
    return fig, win_prob, median_pnl, worst_case_pnl

# ---------------------------------------------------------
# 4. Main App & UI Rendering
# ---------------------------------------------------------
def main():
    st.title("📈 Quant Strategy Dashboard")
    st.markdown("Compare multiple backtest results and deep-dive into performance metrics.")
    
    if 'datasets' not in st.session_state: st.session_state.datasets = {}
        
    col1, col2 = st.columns([3, 1])
    with col2:
        uploaded_files = st.file_uploader("Upload TradingView CSVs", type=['csv'], accept_multiple_files=True)
        if uploaded_files:
            for i, file in enumerate(uploaded_files):
                if file.name not in st.session_state.datasets:
                    color = STRATEGY_COLORS[len(st.session_state.datasets) % len(STRATEGY_COLORS)]
                    dataset = process_tradingview_csv(file, color)
                    if dataset: st.session_state.datasets[file.name] = dataset
            st.success(f"Loaded {len(st.session_state.datasets)} strategies.")
            
    if not st.session_state.datasets:
        st.info("👆 Please upload one or more CSV files from TradingView to begin.")
        return

    datasets = list(st.session_state.datasets.values())
    
    st.header("📊 Strategy Comparison")
    summary_data = []
    for ds in datasets:
        row = {'Strategy Name': ds['name']}
        row.update(ds['stats'])
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(
        df_summary.style.format({
            'Net Profit': "{:,.2f}", 'Max Drawdown': "{:,.2f}", 'Avg Trade': "{:,.2f}",
            'Win Rate (%)': "{:.1f}%", 'Profit Factor': "{:.2f}", 'Sharpe Ratio': "{:.2f}", 'Recovery Factor': "{:.2f}"
        }).map(lambda x: 'color: #10b981' if x > 0 else 'color: #ef4444', subset=['Net Profit']),
        use_container_width=True, height=200
    )
    
    st.subheader("Combined Equity Curves")
    fig_combined = go.Figure()
    for ds in datasets:
        daily_cum = ds['data'].groupby('Date')['PnL'].sum().cumsum()
        fig_combined.add_trace(go.Scatter(x=daily_cum.index, y=daily_cum.values, mode='lines', name=ds['name'], line=dict(color=ds['color'], width=2)))
        
    fig_combined.update_layout(template="plotly_dark", height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified")
    st.plotly_chart(fig_combined, use_container_width=True)
    
    st.divider()
    
    st.header("🔍 Deep Dive Analysis")
    selected_name = st.selectbox("Select Strategy to Analyze:", [ds['name'] for ds in datasets])
    active_ds = next((ds for ds in datasets if ds['name'] == selected_name), None)
    
    if active_ds:
        stats = active_ds['stats']
        df_active = active_ds['data']
        color = active_ds['color']
        top_drawdowns = active_ds['top_drawdowns']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Profit", f"฿ {stats['Net Profit']:,.2f}", f"Avg Trade: ฿ {stats['Avg Trade']:.2f}")
        c2.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}", f"Win Rate: {stats['Win Rate (%)']:.1f}%")
        c3.metric("Max Drawdown", f"฿ {stats['Max Drawdown']:,.2f}", f"Recovery: {stats['Recovery Factor']:.2f}", delta_color="inverse")
        c4.metric("Profit Factor", f"{stats['Profit Factor']:.2f}", f"Cons. Losses: {stats['Max Cons. Loss']}")

       # --- ลบ col_chart1 และ col_chart2 ของเดิมทิ้ง แล้ววางชุดนี้แทน ---
        
        st.markdown("**📈 Equity Curve & Underwater Drawdown (1:1 Scale)**")
        fig_eq_dd = go.Figure()
        
        # แปลงสี Hex เป็น RGBA เพื่อให้กราฟพื้นที่โปร่งแสงสวยงาม
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgba_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)"
        
        # 1. วาดกราฟ Equity Curve (กำไรสะสม)
        fig_eq_dd.add_trace(go.Scatter(
            x=df_active['Trade_Index'], 
            y=df_active['Running_PnL'],
            name="Equity (THB)",
            fill='tozeroy', 
            mode='lines', 
            line=dict(color=color, width=2),
            fillcolor=rgba_color
        ))
        
        # 2. วาดกราฟ Drawdown (จมน้ำ) ทับลงไปในแกน Y เดียวกัน
        fig_eq_dd.add_trace(go.Scatter(
            x=df_active['Trade_Index'], 
            y=df_active['Drawdown'],
            name="Drawdown (THB)",
            fill='tozeroy', 
            mode='lines', 
            line=dict(color='#ef4444', width=1),
            fillcolor='rgba(239, 68, 68, 0.4)' # สีแดงโปร่งแสง
        ))
        
        fig_eq_dd.update_layout(
            template="plotly_dark", 
            height=450, 
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified", # เอาเมาส์ชี้แล้วโชว์ข้อมูล 2 เส้นพร้อมกัน
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq_dd, use_container_width=True) 
            
        # --- NEW: Top 3 Longest Drawdowns Table ---
        st.markdown("### 🕒 Top 3 Longest Drawdown Periods")
        if top_drawdowns:
            df_dd = pd.DataFrame(top_drawdowns)
            df_dd.index = df_dd.index + 1
            st.dataframe(
                df_dd.style.format({'Max Depth (THB)': "฿ {:,.2f}"})
                .map(lambda x: 'color: #ef4444' if isinstance(x, (int, float)) and x < 0 else ''),
                use_container_width=True
            )
        else:
            st.info("No drawdown periods found! (Perfect strategy?)")

        with st.expander("🎲 Run Monte Carlo Simulation (Stress Test)"):
            st.write("จำลองสลับลำดับการเกิดกำไร/ขาดทุน (Resampling) จำนวน 1,000 รูปแบบ เพื่อทดสอบความทนทานของกลยุทธ์")
            if st.button("Run Simulation 🚀"):
                with st.spinner("Calculating 1,000 futures..."):
                    mc_fig, win_prob, median_pnl, worst_case = run_monte_carlo(df_active)
                    mc_c1, mc_c2, mc_c3 = st.columns(3)
                    mc_c1.metric("Probability of Profitability", f"{win_prob:.1f}%")
                    mc_c2.metric("Median Simulated P&L", f"฿ {median_pnl:,.2f}")
                    mc_c3.metric("Worst Case (5th Percentile)", f"฿ {worst_case:,.2f}", delta_color="inverse")
                    st.plotly_chart(mc_fig, use_container_width=True)

if __name__ == "__main__":
    main()
