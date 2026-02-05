import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
from pathlib import Path
import zipfile
from io import StringIO

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Freqtrade Backtest Dashboard",
    layout="wide",
)

BACKTEST_DIR = Path("user_data/backtest_results")
HYPEROPT_DIR = Path("user_data/hyperopt_results")

st.title("📊 Freqtrade Backtest & Hyperopt Dashboard")
st.caption("Simple, human-readable analytics for Freqtrade backtests.")

# make sure they exist
backtest_file = None
hyperopt_file = None


# ---------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------
def get_server_files(folder: Path, allowed_suffixes):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed_suffixes],
        key=lambda p: p.name,
    )


# ---------------------------------------------------------
# TRADE DETECTION HELPERS
# ---------------------------------------------------------
def looks_like_trades_df(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    profit_cols = {"profit_abs", "profit_ratio", "profit_percent", "profit", "profit_usd"}
    date_cols = {"close_date", "open_date", "date", "open_time", "close_time"}
    return bool(cols & profit_cols) and bool(cols & date_cols)


def recursive_find_trades(obj):
    """
    Recursively walk a JSON-like object to find a list of trade dicts.
    Works for many Freqtrade export formats.
    """
    # Direct list of dicts
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        df = pd.DataFrame(obj)
        if looks_like_trades_df(df):
            return obj

    # Dict with 'trades' key or nested structures
    if isinstance(obj, dict):
        if "trades" in obj and isinstance(obj["trades"], list):
            if obj["trades"] and isinstance(obj["trades"][0], dict):
                df = pd.DataFrame(obj["trades"])
                if looks_like_trades_df(df):
                    return obj["trades"]

        for v in obj.values():
            res = recursive_find_trades(v)
            if res is not None:
                return res

    # List of nested things
    if isinstance(obj, list):
        for el in obj:
            res = recursive_find_trades(el)
            if res is not None:
                return res

    return None


def finalize_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up and standardize the trades dataframe.
    """
    df = df.copy()

    # Parse dates
    for col in [
        "open_date",
        "close_date",
        "date",
        "open_time",
        "close_time",
        "open_date_utc",
        "close_date_utc",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure close_date exists
    if "close_date" not in df.columns:
        for candidate in ["date", "close_time"]:
            if candidate in df.columns:
                df["close_date"] = df[candidate]
                break

    # Profit normalization
    if "profit_abs" not in df.columns:
        for candidate in ["profit", "profit_usd"]:
            if candidate in df.columns:
                df["profit_abs"] = df[candidate]
                break

    if "profit_percent" not in df.columns and "profit_ratio" in df.columns:
        df["profit_percent"] = df["profit_ratio"] * 100.0

    return df


def estimate_initial_capital(trades: pd.DataFrame) -> float:
    """
    Rough but decent estimate of starting capital:
    - If stake_amount is available: assume fixed stake and use first stake.
    - Otherwise: use 10x median absolute profit as a fallback.
    """
    if "stake_amount" in trades.columns:
        stake_series = trades["stake_amount"].dropna()
        if not stake_series.empty:
            return float(stake_series.iloc[0])

    # fallback
    median_p = trades["profit_abs"].abs().median() if "profit_abs" in trades.columns else 100.0
    return max(100.0, float(median_p * 10))


# ---------------------------------------------------------
# LOAD BACKTEST (JSON / CSV / ZIP)
# ---------------------------------------------------------
@st.cache_data
def load_backtest(src):
    if src is None:
        return None, {}

    # ---------- PATH ON DISK ----------
    if isinstance(src, (str, Path)):
        path = Path(src)

        # CSV
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            return finalize_trades_df(df), {}

        # ZIP
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                json_files = [n for n in z.namelist() if n.lower().endswith(".json")]
                for name in json_files:
                    with z.open(name) as f:
                        data = json.load(f)
                    trades_list = recursive_find_trades(data)
                    if trades_list:
                        df = pd.DataFrame(trades_list)
                        return finalize_trades_df(df), {}
            return None, {}

        # Plain JSON
        with path.open("r") as f:
            data = json.load(f)
        trades_list = recursive_find_trades(data)
        if trades_list:
            df = pd.DataFrame(trades_list)
            return finalize_trades_df(df), {}
        return None, {}

    # ---------- UPLOADED FILE ----------
    # CSV upload
    if src.name.lower().endswith(".csv"):
        df = pd.read_csv(src)
        return finalize_trades_df(df), {}

    # ZIP upload
    if src.name.lower().endswith(".zip"):
        z = zipfile.ZipFile(src)
        json_files = [n for n in z.namelist() if n.lower().endswith(".json")]
        for name in json_files:
            with z.open(name) as f:
                data = json.load(f)
            trades_list = recursive_find_trades(data)
            if trades_list:
                df = pd.DataFrame(trades_list)
                return finalize_trades_df(df), {}
        return None, {}

    # JSON upload
    data = json.load(src)
    trades_list = recursive_find_trades(data)
    if trades_list:
        df = pd.DataFrame(trades_list)
        return finalize_trades_df(df), {}
    return None, {}


# ---------------------------------------------------------
# LOAD HYPEROPT
# ---------------------------------------------------------
@st.cache_data
def load_hyperopt(src):
    if src is None:
        return None

    if isinstance(src, (str, Path)):
        path = Path(src)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        with path.open("r") as f:
            data = json.load(f)
    else:
        if src.name.lower().endswith(".csv"):
            return pd.read_csv(src)
        data = json.load(src)

    if isinstance(data, dict) and "results" in data:
        return pd.json_normalize(data["results"])
    return pd.json_normalize(data)


# ---------------------------------------------------------
# EQUITY & RISK HELPERS
# ---------------------------------------------------------
def compute_equity_curve(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    df = trades.sort_values("close_date").copy()
    df["equity"] = initial_capital + df["profit_abs"].cumsum()
    df["running_max"] = df["equity"].cummax()
    df["drawdown_pct"] = df["equity"] / df["running_max"] - 1
    df["drawdown_pct"] = df["drawdown_pct"].fillna(0.0)
    return df


def compute_trade_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "open_date" in df.columns and "close_date" in df.columns:
        df["duration_min"] = (df["close_date"] - df["open_date"]).dt.total_seconds() / 60.0
    else:
        df["duration_min"] = np.nan
    return df


def compute_sharpe(equity_df: pd.DataFrame) -> float:
    if "close_date" not in equity_df.columns:
        return np.nan
    s = equity_df.set_index("close_date")["equity"].sort_index()
    daily = s.resample("1D").last().ffill()
    rets = daily.pct_change().dropna()
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return np.nan
    sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(252)
    return float(sharpe)


def compute_risk_stats(trades: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float):
    results = {}
    profits = trades["profit_abs"]

    # max / min trade
    results["max_win"] = float(profits.max())
    results["max_loss"] = float(profits.min())
    losses = profits[profits < 0]
    results["median_loss"] = float(losses.median()) if not losses.empty else np.nan

    # longest losing streak
    streak = 0
    max_streak = 0
    max_streak_start = None
    max_streak_end = None
    current_start = None

    for idx, p in profits.items():
        if p <= 0:
            streak += 1
            if current_start is None:
                current_start = idx
            if streak > max_streak:
                max_streak = streak
                max_streak_start = current_start
                max_streak_end = idx
        else:
            streak = 0
            current_start = None

    results["max_losing_streak_trades"] = int(max_streak)
    if max_streak > 0 and "close_date" in trades.columns:
        start_date = trades.loc[max_streak_start, "close_date"]
        end_date = trades.loc[max_streak_end, "close_date"]
        results["max_losing_streak_days"] = float((end_date - start_date).days)
    else:
        results["max_losing_streak_days"] = 0.0

    # drawdown stats
    results["max_drawdown_pct"] = float(equity_df["drawdown_pct"].min()) * 100.0

    # Monte Carlo risk of 20%+ drawdown
    trade_rets = profits / initial_capital
    trade_rets = trade_rets.values
    if len(trade_rets) < 5:
        results["prob_dd_20"] = np.nan
    else:
        runs = 1000
        n = len(trade_rets)
        ruin_count = 0
        for _ in range(runs):
            seq = np.random.choice(trade_rets, size=n, replace=True)
            eq = initial_capital + np.cumsum(seq * initial_capital)
            running_max = np.maximum.accumulate(eq)
            dd = eq / running_max - 1
            if dd.min() <= -0.2:
                ruin_count += 1
        results["prob_dd_20"] = ruin_count / runs * 100.0

    return results


# ---------------------------------------------------------
# STRATEGY SUMMARY & TEXT INSIGHTS
# ---------------------------------------------------------
def build_summary_metrics(trades: pd.DataFrame):
    total_trades = len(trades)
    wins = int((trades["profit_abs"] > 0).sum())
    winrate = wins / total_trades * 100.0 if total_trades > 0 else 0.0
    total_profit = float(trades["profit_abs"].sum())

    initial_capital = estimate_initial_capital(trades)
    equity_df = compute_equity_curve(trades, initial_capital)
    equity_start = initial_capital
    equity_end = float(equity_df["equity"].iloc[-1])
    total_return = equity_end / equity_start - 1.0

    period_days = (
        (trades["close_date"].max() - trades["close_date"].min()).days
        if "close_date" in trades.columns
        else 0
    )
    if period_days <= 0:
        annual_return = np.nan
    else:
        annual_return = (1.0 + total_return) ** (365.0 / period_days) - 1.0

    sharpe = compute_sharpe(equity_df)

    return {
        "total_trades": total_trades,
        "wins": wins,
        "winrate": winrate,
        "total_profit": total_profit,
        "initial_capital": initial_capital,
        "equity_df": equity_df,
        "total_return_pct": total_return * 100.0,
        "annual_return_pct": annual_return * 100.0 if not np.isnan(annual_return) else np.nan,
        "sharpe": sharpe,
    }


def build_text_insight(trades: pd.DataFrame, metrics: dict, risk_stats: dict) -> str:
    """
    Generate a simple natural-language explanation for beginners.
    """
    lines = []

    # Overall verdict
    if metrics["total_return_pct"] > 0:
        lines.append(
            f"- Overall the strategy **made money**, with a total return of about "
            f"**{metrics['total_return_pct']:.1f}%** over the tested period."
        )
    else:
        lines.append(
            f"- Overall the strategy **lost money**, with a total return of about "
            f"**{metrics['total_return_pct']:.1f}%** over the tested period."
        )

    # Risk/Sharpe
    if not np.isnan(metrics["sharpe"]):
        if metrics["sharpe"] >= 2:
            sharpe_comment = "excellent risk-adjusted performance."
        elif metrics["sharpe"] >= 1:
            sharpe_comment = "decent risk-adjusted performance."
        else:
            sharpe_comment = "weak risk-adjusted performance (returns are small relative to volatility)."
        lines.append(
            f"- Sharpe ratio is **{metrics['sharpe']:.2f}**, which indicates {sharpe_comment}"
        )

    # Drawdown
    dd = risk_stats["max_drawdown_pct"]
    if dd <= -40:
        lines.append(
            f"- Max drawdown is about **{dd:.1f}%**, which is quite deep. "
            "You may want to reduce position size or add stricter risk controls."
        )
    elif dd <= -25:
        lines.append(
            f"- Max drawdown is about **{dd:.1f}%**, which is significant but may be acceptable "
            "if you are comfortable with higher risk."
        )
    else:
        lines.append(
            f"- Max drawdown is about **{dd:.1f}%**, which is relatively moderate for a trading strategy."
        )

    # Pairs
    if "pair" in trades.columns:
        pair_profit = trades.groupby("pair")["profit_abs"].sum().sort_values(ascending=False)
        best_pair = pair_profit.index[0]
        best_pair_val = pair_profit.iloc[0]
        lines.append(
            f"- The best-performing pair was **{best_pair}**, contributing roughly "
            f"**{best_pair_val:.2f}** units of profit."
        )
        if len(pair_profit) > 1:
            worst_pair = pair_profit.index[-1]
            worst_pair_val = pair_profit.iloc[-1]
            lines.append(
                f"- The weakest pair was **{worst_pair}**, dragging results by about "
                f"**{worst_pair_val:.2f}** units."
            )

    # Holding time
    if "trade_duration" in trades.columns:
        med_dur = trades["trade_duration"].median()
        lines.append(
            f"- Median holding time is about **{med_dur:.0f} minutes**, so this strategy "
            "is closer to swing trading than high-frequency scalping."
        )
    elif "duration_min" in trades.columns:
        med_dur = trades["duration_min"].median()
        lines.append(
            f"- Median holding time is about **{med_dur:.0f} minutes**."
        )

    # Risk-of-20% DD
    if not np.isnan(risk_stats["prob_dd_20"]):
        lines.append(
            f"- Based on a simple Monte Carlo simulation, there's roughly a "
            f"**{risk_stats['prob_dd_20']:.0f}%** chance of seeing a **20% or worse drawdown** "
            "over a similar number of trades."
        )

    return "\n".join(lines)


# ---------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------
def build_html_report(trades: pd.DataFrame, metrics: dict, risk_stats: dict) -> str:
    buf = StringIO()
    buf.write("<html><head><meta charset='utf-8'><title>Freqtrade Backtest Report</title></head><body>")
    buf.write("<h1>Freqtrade Backtest Report</h1>")

    buf.write("<h2>Summary</h2>")
    buf.write("<ul>")
    buf.write(f"<li>Total trades: {metrics['total_trades']}</li>")
    buf.write(f"<li>Win rate: {metrics['winrate']:.2f}%</li>")
    buf.write(f"<li>Total profit (stake): {metrics['total_profit']:.4f}</li>")
    buf.write(f"<li>Approx. total return: {metrics['total_return_pct']:.2f}%</li>")
    if not np.isnan(metrics["annual_return_pct"]):
        buf.write(f"<li>Approx. annualized return: {metrics['annual_return_pct']:.2f}%</li>")
    if not np.isnan(metrics["sharpe"]):
        buf.write(f"<li>Sharpe ratio: {metrics['sharpe']:.2f}</li>")
    buf.write(f"<li>Max drawdown: {risk_stats['max_drawdown_pct']:.2f}%</li>")
    buf.write("</ul>")

    buf.write("<h2>Risk</h2>")
    buf.write("<ul>")
    buf.write(f"<li>Max single win: {risk_stats['max_win']:.4f}</li>")
    buf.write(f"<li>Max single loss: {risk_stats['max_loss']:.4f}</li>")
    buf.write(f"<li>Median loss: {risk_stats['median_loss']:.4f}</li>")
    buf.write(f"<li>Longest losing streak: {risk_stats['max_losing_streak_trades']} trades</li>")
    buf.write(f"<li>Longest losing streak length: {risk_stats['max_losing_streak_days']:.0f} days</li>")
    if not np.isnan(risk_stats["prob_dd_20"]):
        buf.write(f"<li>Prob. of 20%+ drawdown (MC): {risk_stats['prob_dd_20']:.0f}%</li>")
    buf.write("</ul>")

    buf.write("<h2>Sample trades</h2>")
    buf.write(trades.head(20).to_html(index=False))

    buf.write("</body></html>")
    return buf.getvalue()


# =========================================================
# SIDEBAR FILE SELECTORS
# =========================================================
st.sidebar.subheader("Backtest file")
bt_mode = st.sidebar.radio(
    "Backtest source",
    ["Pick from server", "Upload file"],
    index=0,
    key="bt_mode",
)

if bt_mode == "Upload file":
    backtest_file = st.sidebar.file_uploader(
        "Upload backtest file",
        type=["json", "csv", "zip"],
        key="bt_upload",
    )
else:
    bt_files = get_server_files(BACKTEST_DIR, [".json", ".csv", ".zip"])
    bt_names = ["(none)"] + [p.name for p in bt_files]
    bt_choice = st.sidebar.selectbox("Select backtest file", bt_names, key="bt_select")
    if bt_choice != "(none)":
        backtest_file = BACKTEST_DIR / bt_choice

st.sidebar.subheader("Hyperopt file")
ho_mode = st.sidebar.radio(
    "Hyperopt source",
    ["Pick from server", "Upload file"],
    index=0,
    key="ho_mode",
)

if ho_mode == "Upload file":
    hyperopt_file = st.sidebar.file_uploader(
        "Upload hyperopt file",
        type=["json", "csv"],
        key="ho_upload",
    )
else:
    ho_files = get_server_files(HYPEROPT_DIR, [".json", ".csv"])
    ho_names = ["(none)"] + [p.name for p in ho_files]
    ho_choice = st.sidebar.selectbox("Select hyperopt file", ho_names, key="ho_select")
    if ho_choice != "(none)":
        hyperopt_file = HYPEROPT_DIR / ho_choice

# Strategy comparison files
st.sidebar.subheader("Compare strategies (optional)")
all_bt_files = get_server_files(BACKTEST_DIR, [".json", ".csv", ".zip"])
compare_choices = st.sidebar.multiselect(
    "Additional backtest files to compare",
    [p.name for p in all_bt_files],
    default=[],
    key="compare_select",
)

st.sidebar.markdown("---")
st.sidebar.write("Backtests live in: `user_data/backtest_results`")
st.sidebar.write("Hyperopt results: `user_data/hyperopt_results`")


# =========================================================
# MAIN TABS
# =========================================================
tabs = st.tabs([
    "📌 Overview",
    "📈 Equity & Drawdown",
    "📍 Trade Scatter",
    "🔥 Heatmaps",
    "🧪 Hyperopt Explorer",
])


# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------
with tabs[0]:
    st.header("📌 Overview")

    if backtest_file is None:
        st.info("Select a backtest file from the sidebar.")
    else:
        trades, _ = load_backtest(backtest_file)

        if trades is None or trades.empty:
            st.warning("No trades found in backtest file.")
        else:
            metrics = build_summary_metrics(trades)
            risk_stats = compute_risk_stats(trades, metrics["equity_df"], metrics["initial_capital"])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total trades", metrics["total_trades"])
            col2.metric("Win rate", f"{metrics['winrate']:.1f}%", help="Percentage of trades that were profitable.")
            col3.metric("Total profit (stake)", f"{metrics['total_profit']:.2f}")
            col4.metric(
                "Approx. total return",
                f"{metrics['total_return_pct']:.1f}%",
                help="Based on initial capital estimated from stake_amount.",
            )

            col5, col6, col7 = st.columns(3)
            if not np.isnan(metrics["annual_return_pct"]):
                col5.metric(
                    "Approx. annualized return",
                    f"{metrics['annual_return_pct']:.1f}%",
                    help="Normalizes performance to a 1-year period.",
                )
            else:
                col5.metric("Approx. annualized return", "n/a")
            if not np.isnan(metrics["sharpe"]):
                col6.metric(
                    "Sharpe ratio",
                    f"{metrics['sharpe']:.2f}",
                    help="Return divided by volatility. >1 is decent, >2 is strong.",
                )
            else:
                col6.metric("Sharpe ratio", "n/a")
            col7.metric(
                "Max drawdown",
                f"{risk_stats['max_drawdown_pct']:.1f}%",
                help="Worst peak-to-trough fall in equity during the test.",
            )

            st.subheader("Plain-English insight")
            st.markdown(build_text_insight(trades, metrics, risk_stats))

            st.subheader("Sample of trades")
            st.dataframe(trades.head(20))

            # Downloadable HTML report
            st.subheader("Export report")
            html_report = build_html_report(trades, metrics, risk_stats)
            st.download_button(
                "📥 Download HTML report (for PDF printing)",
                data=html_report,
                file_name="backtest_report.html",
                mime="text/html",
            )


# ---------------------------------------------------------
# TAB 2: EQUITY & DRAWDOWN (+ COMPARISON & RISK)
# ---------------------------------------------------------
with tabs[1]:
    st.header("📈 Equity & Drawdown")

    if backtest_file is None:
        st.info("Select a backtest file first.")
    else:
        trades, _ = load_backtest(backtest_file)
        if trades is None or trades.empty:
            st.warning("No trades found in backtest file.")
        else:
            metrics = build_summary_metrics(trades)
            equity_df = metrics["equity_df"]
            risk_stats = compute_risk_stats(trades, equity_df, metrics["initial_capital"])

            st.subheader("Equity curve")
            fig_eq = px.line(
                equity_df,
                x="close_date",
                y="equity",
                title="Equity (cumulative profit, assuming 1x stake as starting capital)",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("Underwater (drawdown %)")
            fig_dd = px.area(
                equity_df,
                x="close_date",
                y="drawdown_pct",
                title="Drawdown %",
            )
            fig_dd.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_dd, use_container_width=True)

            st.subheader("Risk analytics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Max single win", f"{risk_stats['max_win']:.2f}")
            c2.metric("Max single loss", f"{risk_stats['max_loss']:.2f}")
            c3.metric("Median loss", f"{risk_stats['median_loss']:.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Longest losing streak (trades)", f"{risk_stats['max_losing_streak_trades']}")
            c5.metric("Longest losing streak (days)", f"{risk_stats['max_losing_streak_days']:.0f}")
            if not np.isnan(risk_stats["prob_dd_20"]):
                c6.metric(
                    "Prob(DD ≤ -20%)",
                    f"{risk_stats['prob_dd_20']:.0f}%",
                    help="Chance of a 20% or worse drawdown over a similar number of trades (Monte Carlo estimate).",
                )
            else:
                c6.metric("Prob(DD ≤ -20%)", "n/a")

            # Strategy comparison
            if compare_choices:
                st.subheader("Equity comparison with other backtests")
                curves = []
                names = []

                for fname in compare_choices:
                    path = BACKTEST_DIR / fname
                    tdf, _ = load_backtest(path)
                    if tdf is None or tdf.empty:
                        continue
                    init_cap = estimate_initial_capital(tdf)
                    eq = compute_equity_curve(tdf, init_cap)
                    eq["strategy"] = fname
                    curves.append(eq[["close_date", "equity", "strategy"]])
                    names.append(fname)

                if curves:
                    all_curves = pd.concat(curves, ignore_index=True)
                    fig_cmp = px.line(
                        all_curves,
                        x="close_date",
                        y="equity",
                        color="strategy",
                        title="Equity curves comparison",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)
                else:
                    st.info("No valid additional backtest files selected for comparison.")


# ---------------------------------------------------------
# TAB 3: TRADE SCATTER
# ---------------------------------------------------------
with tabs[2]:
    st.header("📍 Trade Scatter")

    if backtest_file is None:
        st.info("Select a backtest file first.")
    else:
        trades, _ = load_backtest(backtest_file)
        if trades is None or trades.empty:
            st.warning("No trades found in backtest file.")
        else:
            trades = compute_trade_duration(trades)
            y_opts = [c for c in ["profit_abs", "profit_percent"] if c in trades.columns]
            if not y_opts:
                st.warning("No profit columns found.")
            else:
                y_col = st.selectbox("Y-axis", y_opts)
                color_col = "pair" if "pair" in trades.columns else None
                st.caption("Each dot is one trade. X = holding time, Y = profit.")
                fig_sc = px.scatter(
                    trades,
                    x="duration_min",
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} vs duration (minutes)",
                )
                st.plotly_chart(fig_sc, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: HEATMAPS (PLUS BEST/WORST MONTHS & PAIRS)
# ---------------------------------------------------------
with tabs[3]:
    st.header("🔥 Heatmaps")

    if backtest_file is None:
        st.info("Select a backtest file first.")
    else:
        trades, _ = load_backtest(backtest_file)
        if trades is None or trades.empty:
            st.warning("No trades found in backtest file.")
        else:
            if "pair" not in trades.columns or "close_date" not in trades.columns:
                st.warning("Need 'pair' and 'close_date' columns to build heatmaps.")
            else:
                trades["month"] = trades["close_date"].dt.to_period("M").astype(str)
                trades["weekday"] = trades["close_date"].dt.day_name()

                metric = st.selectbox(
                    "Metric",
                    [c for c in ["profit_abs", "profit_percent"] if c in trades.columns],
                    index=0,
                )

                st.subheader("Profit by pair & month")
                pivot_month = trades.pivot_table(
                    index="pair",
                    columns="month",
                    values=metric,
                    aggfunc="sum",
                    fill_value=0,
                )
                fig_hm1 = px.imshow(
                    pivot_month,
                    labels=dict(x="Month", y="Pair", color=metric),
                    aspect="auto",
                )
                st.plotly_chart(fig_hm1, use_container_width=True)

                # Best / worst months
                month_profit = (
                    trades.groupby("month")["profit_abs"].sum().sort_values(ascending=False)
                )
                st.subheader("Best / worst months (by total profit)")
                c1, c2 = st.columns(2)
                c1.write("**Best months**")
                c1.table(month_profit.head(3))
                c2.write("**Worst months**")
                c2.table(month_profit.tail(3))

                # Best / worst pairs
                pair_profit = (
                    trades.groupby("pair")["profit_abs"].sum().sort_values(ascending=False)
                )
                st.subheader("Best / worst pairs (by total profit)")
                c3, c4 = st.columns(2)
                c3.write("**Best pairs**")
                c3.table(pair_profit.head(3))
                c4.write("**Worst pairs**")
                c4.table(pair_profit.tail(3))


# ---------------------------------------------------------
# TAB 5: HYPEROPT EXPLORER
# ---------------------------------------------------------
with tabs[4]:
    st.header("🧪 Hyperopt Explorer")

    if hyperopt_file is None:
        st.info("Select or upload a hyperopt file.")
    else:
        hdf = load_hyperopt(hyperopt_file)
        if hdf is None or hdf.empty:
            st.warning("No hyperopt data loaded.")
        else:
            st.subheader("Raw hyperopt results")
            st.dataframe(hdf)

            num_cols = [c for c in hdf.columns if pd.api.types.is_numeric_dtype(hdf[c])]
            if num_cols:
                sort_col = st.selectbox("Sort by metric", num_cols)
                ascending = st.checkbox("Ascending (e.g. for loss)", value=True)
                sorted_df = hdf.sort_values(sort_col, ascending=ascending)
                st.subheader("Top 20 configurations")
                st.dataframe(sorted_df.head(20))

                st.subheader("Best configuration (top row)")
                st.json(sorted_df.iloc[0].to_dict())
            else:
                st.warning("No numeric columns found in hyperopt data.")

