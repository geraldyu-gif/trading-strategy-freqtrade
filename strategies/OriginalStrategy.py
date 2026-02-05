# pragma: no cover
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
)
from freqtrade.persistence import Trade


class BtcRegimeStrat(IStrategy):
    """
    Long-only breakout / momentum strategy with BTC 4h market regime filter.

    Coin-level logic (1h):
        - ADX + DI uptrend filter
        - Donchian breakout (highest high)
        - Quote-volume expansion
        - True-range (TR) expansion vs ATR
        - EMA fast > EMA slow + price above EMA fast
        - ATR-based dynamic stoploss

    BTC regime (4h, informative pair):
        - BTC EMA fast > EMA slow
        - BTC ADX above threshold
        - BTC +DI - -DI above margin
        - BTC RSI above threshold

    BTC regime parameters are hyperoptable.
    """

    timeframe = "1h"
    startup_candle_count = 200

    can_short = False  # Crypto.com spot, long-only

    # ROI table (already optimized for your long side)
    minimal_roi = {
        "0": 0.522,
        "395": 0.12,
        "821": 0.07,
        "2229": 0
    }

    stoploss = -0.30
    use_custom_stoploss = True
    trailing_stop = False

    # === BTC informative pair settings ===
    btc_pair = "BTC/USDT"
    btc_tf = "4h"

    # ===== Coin-level hyperopt parameters (frozen at good values) =====

    adx_threshold   = IntParameter(25, 45, default=38, space="buy", optimize=False)
    atr_period      = IntParameter(10, 21, default=19, space="buy", optimize=False)
    di_margin       = IntParameter(3, 15, default=15, space="buy", optimize=False)
    donchian_window = IntParameter(30, 80, default=76, space="buy", optimize=False)

    vol_lookback = IntParameter(20, 60, default=38, space="buy", optimize=False)
    vol_mult = DecimalParameter(1.2, 3.0, decimals=1, default=1.8, space="buy", optimize=False)

    tr_mult = DecimalParameter(0.8, 2.0, decimals=1, default=1.8, space="buy", optimize=False)

    ema_fast_period = IntParameter(8, 21, default=8, space="buy", optimize=False)
    ema_slow_period = IntParameter(20, 55, default=28, space="buy", optimize=False)

    # ===== BTC regime params (already hyperopted) =====
    btc_adx_threshold = IntParameter(10, 30, default=10, space="buy", optimize=False)
    btc_di_margin     = IntParameter(0, 10, default=9,  space="buy", optimize=False)
    btc_rsi_threshold = IntParameter(45, 60, default=56, space="buy", optimize=False)

    # ===== ATR stoploss multiplier =====
    atr_stoploss_mult = DecimalParameter(2.5, 6.0, decimals=1, default=5.0,
                                         space="sell", optimize=False)

    # ------------------------------------------------------------------ #
    # Informative pairs
    # ------------------------------------------------------------------ #

    def informative_pairs(self):
        """
        Use BTC/USDT 4h as an informative pair for market regime.
        """
        return [(self.btc_pair, self.btc_tf)]

    # ------------------------------------------------------------------ #
    # Indicators
    # ------------------------------------------------------------------ #

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:

        # -------- Coin-level (1h) indicators --------

        # ADX & DI
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        # EMAs
        ema_fast = int(self.ema_fast_period.value)
        ema_slow = int(self.ema_slow_period.value)
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=ema_slow)

        # Donchian high (on highs)
        donchian_n = int(self.donchian_window.value)
        dataframe["donchian_high"] = (
            dataframe["high"]
            .rolling(window=donchian_n, min_periods=donchian_n)
            .max()
            .shift(1)
        )

        # Quote volume
        dataframe["quote_volume"] = dataframe["close"] * dataframe["volume"]
        vol_n = int(self.vol_lookback.value)
        dataframe["quote_vol_sma"] = (
            dataframe["quote_volume"]
            .rolling(window=vol_n, min_periods=vol_n)
            .mean()
        )

        # ATR & True Range
        atr_n = int(self.atr_period.value)
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"],
            timeperiod=atr_n
        )

        prev_close = dataframe["close"].shift(1)
        tr1 = dataframe["high"] - dataframe["low"]
        tr2 = (dataframe["high"] - prev_close).abs()
        tr3 = (dataframe["low"] - prev_close).abs()
        dataframe["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # -------- BTC 4h regime indicators (informative) --------

        btc_df = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=self.btc_tf)

        btc_df["btc_close"] = btc_df["close"]
        btc_df["btc_ema_fast"] = ta.EMA(btc_df, timeperiod=21)
        btc_df["btc_ema_slow"] = ta.EMA(btc_df, timeperiod=55)

        btc_df["btc_adx"] = ta.ADX(btc_df, timeperiod=14)
        btc_df["btc_plus_di"] = ta.PLUS_DI(btc_df, timeperiod=14)
        btc_df["btc_minus_di"] = ta.MINUS_DI(btc_df, timeperiod=14)

        btc_df["btc_rsi"] = ta.RSI(btc_df, timeperiod=14)

        btc_df["btc_atr"] = ta.ATR(
            btc_df["high"], btc_df["low"], btc_df["close"],
            timeperiod=14
        )

        btc_df = btc_df[
            [
                "date",
                "btc_close",
                "btc_ema_fast",
                "btc_ema_slow",
                "btc_adx",
                "btc_plus_di",
                "btc_minus_di",
                "btc_rsi",
                "btc_atr",
            ]
        ]

        # Merge BTC 4h into 1h dataframe (suffix "_4h")
        dataframe = merge_informative_pair(
            dataframe,
            btc_df,
            self.timeframe,
            self.btc_tf,
            ffill=True,
        )

        return dataframe

    # ------------------------------------------------------------------ #
    # Entry logic (long-only, BTC regime filtered)
    # ------------------------------------------------------------------ #

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:

        # Always reset buy column
        dataframe["buy"] = 0

        required_cols = [
            "adx", "plus_di", "minus_di",
            "donchian_high",
            "quote_volume", "quote_vol_sma",
            "ema_fast", "ema_slow",
            "atr", "true_range",
        ]
        for col in required_cols:
            if col not in dataframe.columns:
                return dataframe

        buy_conditions: List = []

        # Basic sanity
        buy_conditions.append(dataframe["volume"] > 0)
        buy_conditions.append(dataframe["donchian_high"].notna())
        buy_conditions.append(dataframe["quote_vol_sma"].notna())
        buy_conditions.append(dataframe["ema_fast"].notna())
        buy_conditions.append(dataframe["ema_slow"].notna())
        buy_conditions.append(dataframe["atr"].notna())
        buy_conditions.append(dataframe["true_range"].notna())

        # Coin-level trend regime
        buy_conditions.append(dataframe["adx"] > self.adx_threshold.value)
        buy_conditions.append(dataframe["plus_di"] > dataframe["minus_di"])
        buy_conditions.append(
            (dataframe["plus_di"] - dataframe["minus_di"]) > self.di_margin.value
        )

        # Breakout above Donchian high
        buy_conditions.append(dataframe["close"] > dataframe["donchian_high"])

        # Volume expansion
        buy_conditions.append(
            dataframe["quote_volume"] >
            dataframe["quote_vol_sma"] * self.vol_mult.value
        )

        # Volatility expansion
        buy_conditions.append(
            dataframe["true_range"] >
            dataframe["atr"] * self.tr_mult.value
        )

        # EMA trend confirmation
        buy_conditions.append(dataframe["ema_fast"] > dataframe["ema_slow"])
        buy_conditions.append(dataframe["close"] > dataframe["ema_fast"])

        if len(dataframe) > 0:
            print("DEBUG params:",
                  self.adx_threshold.value,
                  self.atr_period.value,
                  self.donchian_window.value,
                  self.btc_adx_threshold.value,
                  self.btc_rsi_threshold.value)
        # only once
        dataframe["debug_printed"] = True


        # --- BTC regime filter (if informative columns exist) ---

        btc_cols = [
            "btc_ema_fast_4h", "btc_ema_slow_4h",
            "btc_adx_4h", "btc_plus_di_4h", "btc_minus_di_4h",
            "btc_rsi_4h",
        ]
        btc_available = all(col in dataframe.columns for col in btc_cols)

        if btc_available:
            btc_valid = (
                dataframe["btc_ema_fast_4h"].notna() &
                dataframe["btc_ema_slow_4h"].notna() &
                dataframe["btc_adx_4h"].notna() &
                dataframe["btc_rsi_4h"].notna()
            )

            btc_trend_up = (
                (dataframe["btc_ema_fast_4h"] > dataframe["btc_ema_slow_4h"]) &
                (dataframe["btc_adx_4h"] > self.btc_adx_threshold.value) &
                (
                    (dataframe["btc_plus_di_4h"] - dataframe["btc_minus_di_4h"]) >
                    self.btc_di_margin.value
                ) &
                (dataframe["btc_rsi_4h"] > self.btc_rsi_threshold.value)
            )

            buy_conditions.append(btc_valid & btc_trend_up)

        if buy_conditions:
            dataframe.loc[
                np.logical_and.reduce(buy_conditions),
                "buy"
            ] = 1

        return dataframe

    # ------------------------------------------------------------------ #
    # Exit logic (long-only)
    # ------------------------------------------------------------------ #

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:

        if "sell" not in dataframe.columns:
            dataframe["sell"] = 0

        sell_conditions: List = []
        sell_conditions.append(dataframe["ema_fast"] < dataframe["ema_slow"])
        sell_conditions.append(dataframe["close"] < dataframe["ema_fast"])

        if sell_conditions:
            dataframe.loc[
                np.logical_and.reduce(sell_conditions),
                "sell"
            ] = 1

        return dataframe

    # ------------------------------------------------------------------ #
    # Custom ATR stoploss (long-only)
    # ------------------------------------------------------------------ #

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if dataframe is None or dataframe.empty:
            return 1

        df = dataframe.loc[dataframe["date"] <= current_time]
        if df.empty:
            return 1

        last_candle = df.iloc[-1]
        atr = last_candle.get("atr", None)

        if atr is None or np.isnan(atr):
            return 1

        atr_mult = float(self.atr_stoploss_mult.value)

        stop_price = trade.open_rate - atr_mult * atr
        rel_sl = (stop_price / current_rate) - 1.0  # negative

        rel_sl = max(rel_sl, self.stoploss)

        return float(rel_sl)

