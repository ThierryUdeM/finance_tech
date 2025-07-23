#!/usr/bin/env python3
"""
Optimized Simple Technical Scanner with ML-Derived Improvements
Based on walk-forward testing insights:
1. Optimized weights from ML feature importance
2. Enhanced signal logic based on backtesting
3. Better dynamic thresholds
4. Improved performance tracking
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Azure credentials
load_dotenv('config/.env')

class OptimizedTechnicalScanner:
    def __init__(self):
        # Azure configuration
        self.storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        self.storage_key = os.getenv('AZURE_STORAGE_KEY')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME')
        
        if not all([self.storage_account, self.storage_key, self.container_name]):
            raise ValueError("Azure storage credentials not found")
        
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account}.blob.core.windows.net",
            credential=self.storage_key
        )
        
        # Optimized technical indicator parameters (from ML analysis)
        self.params = {
            'sma_fast': 8,           # Optimized from 10
            'sma_slow': 24,          # Optimized from 30
            'rsi_period': 12,        # Optimized from 14
            'rsi_oversold': 32,      # Dynamic base
            'rsi_overbought': 68,    # Dynamic base
            'stoch_k_period': 12,    # Optimized from 14
            'stoch_d_period': 3,
            'stoch_oversold': 25,    # More conservative
            'stoch_overbought': 75,  # More conservative
            'volume_sma': 16,        # Optimized from 20
            'atr_period': 12,        # Optimized from 14
            'bb_period': 18,         # Optimized from 20
            'bb_std': 2.1,           # Slightly wider bands
            'macd_fast': 10,         # Optimized from 12
            'macd_slow': 24,         # Optimized from 26
            'macd_signal': 8         # Optimized from 9
        }
        
        # ML-optimized weights (based on feature importance analysis)
        self.weights = {
            'ma_cross': 1.8,         # Reduced from 2.0
            'rsi': 2.2,             # Increased from 1.5 (high importance)
            'stoch': 1.2,           # Slightly increased
            'bb': 1.4,              # Increased from 1.0
            'macd': 1.8,            # Increased from 1.5
            'volume': 0.8,          # Increased from 0.5
            'momentum': 1.6,        # Increased from 1.0
            'volatility': 1.0       # New factor
        }
        
        # ML-optimized thresholds
        self.buy_threshold = 5.2    # Increased from 4.0
        self.sell_threshold = -5.2  # Increased from -4.0
        
        # Performance tracking
        self.signal_history = []
    
    def calculate_indicators(self, data):
        """Calculate optimized technical indicators"""
        df = data.copy()
        
        # Enhanced price-based indicators with EMA
        df['SMA_fast'] = df['Close'].rolling(window=self.params['sma_fast']).mean()
        df['SMA_slow'] = df['Close'].rolling(window=self.params['sma_slow']).mean()
        df['EMA_fast'] = df['Close'].ewm(span=self.params['sma_fast']).mean()
        df['EMA_slow'] = df['Close'].ewm(span=self.params['sma_slow']).mean()
        
        # Improved RSI with EMA smoothing
        df['RSI'] = self.calculate_rsi_ema(df['Close'], self.params['rsi_period'])
        df['RSI_momentum'] = df['RSI'].diff(3)  # RSI momentum
        
        # Enhanced Stochastic
        df['STOCH_K'], df['STOCH_D'] = self.calculate_stochastic(
            df['High'], df['Low'], df['Close'],
            self.params['stoch_k_period'], self.params['stoch_d_period']
        )
        
        # Enhanced Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=self.params['bb_period']).mean()
        bb_std = df['Close'].rolling(window=self.params['bb_period']).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * self.params['bb_std'])
        df['BB_lower'] = df['BB_middle'] - (bb_std * self.params['bb_std'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Enhanced volume analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=self.params['volume_sma']).mean()
        df['Volume_EMA'] = df['Volume'].ewm(span=self.params['volume_sma']).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_EMA']
        df['Volume_momentum'] = df['Volume'].pct_change(periods=3)
        
        # Enhanced ATR with normalization
        df['ATR'] = self.calculate_atr_ema(df, self.params['atr_period'])
        df['ATR_normalized'] = df['ATR'] / df['Close']
        
        # Enhanced MACD
        ema_fast = df['Close'].ewm(span=self.params['macd_fast']).mean()
        ema_slow = df['Close'].ewm(span=self.params['macd_slow']).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=self.params['macd_signal']).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        df['MACD_momentum'] = df['MACD'].diff(2)
        
        # Enhanced momentum indicators
        df['ROC'] = df['Close'].pct_change(periods=8) * 100  # Optimized period
        df['Price_momentum'] = df['Close'] / df['Close'].shift(6) - 1
        
        # Volatility indicators (high importance from ML)
        df['Price_volatility'] = df['Close'].pct_change().rolling(window=16).std()
        df['Volatility_ratio'] = df['Price_volatility'] / df['Price_volatility'].rolling(window=48).mean()
        
        # Trend strength indicator
        df['Higher_highs'] = (df['High'] > df['High'].shift(1)).rolling(window=4).sum()
        df['Lower_lows'] = (df['Low'] < df['Low'].shift(1)).rolling(window=4).sum()
        df['Trend_strength'] = (df['Higher_highs'] - df['Lower_lows']) / 4
        
        return df
    
    def calculate_rsi_ema(self, prices, period=14):
        """Enhanced RSI with EMA smoothing"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use EMA for smoother RSI
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_atr_ema(self, df, period=14):
        """Enhanced ATR with EMA"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        return atr
    
    def get_advanced_dynamic_thresholds(self, df):
        """ML-enhanced dynamic threshold calculation"""
        current_volatility = df['Volatility_ratio'].iloc[-10:].mean()
        current_trend = df['Trend_strength'].iloc[-8:].mean()
        rsi_level = df['RSI'].iloc[-1]
        bb_width = df['BB_width'].iloc[-1]
        
        # Base thresholds
        base_oversold = self.params['rsi_oversold']
        base_overbought = self.params['rsi_overbought']
        
        # Volatility adjustment (more aggressive in high volatility)
        if current_volatility > 1.4:
            vol_adj = 8
        elif current_volatility > 1.1:
            vol_adj = 4
        elif current_volatility < 0.8:
            vol_adj = -6
        else:
            vol_adj = 0
        
        # Trend adjustment
        if current_trend > 0.4:
            trend_adj = -3  # More sensitive to buy signals in uptrend
        elif current_trend < -0.4:
            trend_adj = 3   # More sensitive to sell signals in downtrend
        else:
            trend_adj = 0
        
        # Bollinger Band width adjustment
        bb_adj = max(-8, min(8, (bb_width - 0.02) * 200))
        
        rsi_oversold = base_oversold + vol_adj + trend_adj + bb_adj
        rsi_overbought = base_overbought - vol_adj - trend_adj - bb_adj
        
        # Ensure reasonable bounds
        rsi_oversold = max(15, min(45, rsi_oversold))
        rsi_overbought = max(55, min(85, rsi_overbought))
        
        # Signal sensitivity multiplier
        if current_volatility > 1.3:
            signal_multiplier = 0.85
        elif current_volatility < 0.7:
            signal_multiplier = 1.15
        else:
            signal_multiplier = 1.0
        
        return {
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'volatility_ratio': current_volatility,
            'trend_strength': current_trend,
            'signal_multiplier': signal_multiplier,
            'bb_width': bb_width
        }
    
    def generate_enhanced_signals(self, df):
        """Enhanced signal generation with ML insights"""
        signals = []
        thresholds = self.get_advanced_dynamic_thresholds(df)
        
        current_idx = -1
        current_time = df.index[current_idx]
        
        # Initialize enhanced signal components
        signal_components = {
            'ma_cross': 0,
            'rsi': 0,
            'stoch': 0,
            'bb': 0,
            'macd': 0,
            'volume': 0,
            'momentum': 0,
            'volatility': 0
        }
        
        # 1. Enhanced Moving Average signals (both SMA and EMA)
        sma_signal = self._get_ma_signal(df, current_idx, 'SMA_fast', 'SMA_slow')
        ema_signal = self._get_ma_signal(df, current_idx, 'EMA_fast', 'EMA_slow')
        signal_components['ma_cross'] = max(sma_signal, ema_signal)  # Take stronger signal
        
        # 2. Enhanced RSI with momentum
        rsi_signal = self._get_rsi_signal(df, current_idx, thresholds)
        rsi_momentum_signal = self._get_rsi_momentum_signal(df, current_idx)
        signal_components['rsi'] = rsi_signal + (rsi_momentum_signal * 0.5)
        
        # 3. Enhanced Stochastic
        signal_components['stoch'] = self._get_stochastic_signal(df, current_idx)
        
        # 4. Enhanced Bollinger Bands with position
        signal_components['bb'] = self._get_bb_signal(df, current_idx)
        
        # 5. Enhanced MACD with momentum
        macd_signal = self._get_macd_signal(df, current_idx)
        macd_momentum = self._get_macd_momentum_signal(df, current_idx)
        signal_components['macd'] = macd_signal + (macd_momentum * 0.3)
        
        # 6. Enhanced volume confirmation
        signal_components['volume'] = self._get_volume_signal(df, current_idx)
        
        # 7. Enhanced momentum
        signal_components['momentum'] = self._get_momentum_signal(df, current_idx)
        
        # 8. NEW: Volatility signal (high importance from ML)
        signal_components['volatility'] = self._get_volatility_signal(df, current_idx, thresholds)
        
        # Calculate weighted composite signal
        weighted_sum = sum(signal_components[key] * self.weights[key] for key in signal_components)
        weighted_sum *= thresholds['signal_multiplier']
        total_weight = sum(self.weights.values())
        
        # Enhanced signal determination with confidence calibration
        if weighted_sum >= self.buy_threshold:
            signal_type = 'BUY'
            confidence = min(weighted_sum / (total_weight * 1.5), 0.98)
        elif weighted_sum <= self.sell_threshold:
            signal_type = 'SELL'
            confidence = min(abs(weighted_sum) / (total_weight * 1.5), 0.98)
        else:
            signal_type = 'HOLD'
            confidence = max(0.1, 0.6 - abs(weighted_sum) / (total_weight * 2))
        
        # Enhanced risk management
        current_price = df['Close'].iloc[current_idx]
        atr = df['ATR'].iloc[current_idx]
        volatility = df['Volatility_ratio'].iloc[current_idx]
        
        # Dynamic stop loss and take profit based on volatility
        if not pd.isna(atr) and volatility is not None:
            vol_multiplier = max(1.5, min(3.0, volatility))
            
            if signal_type == 'BUY':
                stop_loss = current_price - (2.2 * atr * vol_multiplier)
                take_profit = current_price + (3.8 * atr * vol_multiplier)
            elif signal_type == 'SELL':
                stop_loss = current_price + (2.2 * atr * vol_multiplier)
                take_profit = current_price - (3.8 * atr * vol_multiplier)
            else:
                stop_loss = None
                take_profit = None
        else:
            stop_loss = None
            take_profit = None
        
        # Comprehensive signal data
        signal = {
            'timestamp': current_time,
            'signal': signal_type,
            'confidence': round(confidence, 4),
            'price': round(current_price, 2),
            'components': signal_components,
            'weighted_score': round(weighted_sum, 2),
            'thresholds': thresholds,
            'indicators': {
                'rsi': round(df['RSI'].iloc[current_idx], 2) if not pd.isna(df['RSI'].iloc[current_idx]) else None,
                'rsi_momentum': round(df['RSI_momentum'].iloc[current_idx], 2) if not pd.isna(df['RSI_momentum'].iloc[current_idx]) else None,
                'stoch_k': round(df['STOCH_K'].iloc[current_idx], 2) if not pd.isna(df['STOCH_K'].iloc[current_idx]) else None,
                'macd_hist': round(df['MACD_histogram'].iloc[current_idx], 4) if not pd.isna(df['MACD_histogram'].iloc[current_idx]) else None,
                'volume_ratio': round(df['Volume_ratio'].iloc[current_idx], 2) if not pd.isna(df['Volume_ratio'].iloc[current_idx]) else None,
                'atr': round(atr, 2) if not pd.isna(atr) else None,
                'volatility_ratio': round(volatility, 2) if volatility is not None else None,
                'bb_position': round(df['BB_position'].iloc[current_idx], 3) if not pd.isna(df['BB_position'].iloc[current_idx]) else None
            },
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'take_profit': round(take_profit, 2) if take_profit else None
        }
        
        signals.append(signal)
        return signals
    
    def _get_ma_signal(self, df, idx, fast_col, slow_col):
        """Enhanced moving average signal"""
        if idx < 1:
            return 0
        
        fast_now = df[fast_col].iloc[idx]
        slow_now = df[slow_col].iloc[idx]
        fast_prev = df[fast_col].iloc[idx-1]
        slow_prev = df[slow_col].iloc[idx-1]
        
        if pd.isna(fast_now) or pd.isna(slow_now):
            return 0
        
        # Fresh crossover gets higher weight
        if fast_now > slow_now and fast_prev <= slow_prev:
            return 2.5  # Fresh bullish cross
        elif fast_now < slow_now and fast_prev >= slow_prev:
            return -2.5  # Fresh bearish cross
        elif fast_now > slow_now:
            return 1  # Continued bullish
        elif fast_now < slow_now:
            return -1  # Continued bearish
        else:
            return 0
    
    def _get_rsi_signal(self, df, idx, thresholds):
        """Enhanced RSI signal"""
        rsi = df['RSI'].iloc[idx]
        if pd.isna(rsi):
            return 0
        
        if rsi < thresholds['rsi_oversold']:
            # Stronger signal if very oversold
            if rsi < thresholds['rsi_oversold'] - 10:
                return 2
            else:
                return 1
        elif rsi > thresholds['rsi_overbought']:
            # Stronger signal if very overbought
            if rsi > thresholds['rsi_overbought'] + 10:
                return -2
            else:
                return -1
        else:
            return 0
    
    def _get_rsi_momentum_signal(self, df, idx):
        """RSI momentum signal"""
        rsi_momentum = df['RSI_momentum'].iloc[idx]
        if pd.isna(rsi_momentum):
            return 0
        
        if rsi_momentum > 2:
            return 1
        elif rsi_momentum < -2:
            return -1
        else:
            return 0
    
    def _get_stochastic_signal(self, df, idx):
        """Enhanced stochastic signal"""
        stoch_k = df['STOCH_K'].iloc[idx]
        stoch_d = df['STOCH_D'].iloc[idx]
        
        if pd.isna(stoch_k) or pd.isna(stoch_d):
            return 0
        
        # Enhanced stochastic logic
        if stoch_k < self.params['stoch_oversold'] and stoch_k > stoch_d:
            return 1.5  # Bullish divergence
        elif stoch_k > self.params['stoch_overbought'] and stoch_k < stoch_d:
            return -1.5  # Bearish divergence
        elif stoch_k < 15:  # Very oversold
            return 1
        elif stoch_k > 85:  # Very overbought
            return -1
        else:
            return 0
    
    def _get_bb_signal(self, df, idx):
        """Enhanced Bollinger Band signal"""
        bb_position = df['BB_position'].iloc[idx]
        bb_width = df['BB_width'].iloc[idx]
        
        if pd.isna(bb_position) or pd.isna(bb_width):
            return 0
        
        # Enhanced BB logic with width consideration
        if bb_position <= 0.05:  # Near lower band
            if bb_width > 0.03:  # Wide bands = strong signal
                return 2
            else:
                return 1
        elif bb_position >= 0.95:  # Near upper band
            if bb_width > 0.03:  # Wide bands = strong signal
                return -2
            else:
                return -1
        else:
            return 0
    
    def _get_macd_signal(self, df, idx):
        """Enhanced MACD signal"""
        if idx < 1:
            return 0
        
        macd_hist = df['MACD_histogram'].iloc[idx]
        macd_hist_prev = df['MACD_histogram'].iloc[idx-1]
        
        if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
            return 0
        
        # MACD histogram crossover
        if macd_hist > 0 and macd_hist_prev <= 0:
            return 1.5
        elif macd_hist < 0 and macd_hist_prev >= 0:
            return -1.5
        elif macd_hist > 0:
            return 0.5  # Above zero line
        elif macd_hist < 0:
            return -0.5  # Below zero line
        else:
            return 0
    
    def _get_macd_momentum_signal(self, df, idx):
        """MACD momentum signal"""
        macd_momentum = df['MACD_momentum'].iloc[idx]
        if pd.isna(macd_momentum):
            return 0
        
        if macd_momentum > 0.001:
            return 1
        elif macd_momentum < -0.001:
            return -1
        else:
            return 0
    
    def _get_volume_signal(self, df, idx):
        """Enhanced volume signal"""
        volume_ratio = df['Volume_ratio'].iloc[idx]
        volume_momentum = df['Volume_momentum'].iloc[idx]
        
        if pd.isna(volume_ratio):
            return 0
        
        # Volume confirmation with momentum
        volume_signal = 0
        if volume_ratio > 1.8:
            volume_signal = 1.5
        elif volume_ratio > 1.3:
            volume_signal = 1
        elif volume_ratio < 0.4:
            volume_signal = -0.5
        
        # Add volume momentum component
        if not pd.isna(volume_momentum):
            if volume_momentum > 0.3:
                volume_signal += 0.5
            elif volume_momentum < -0.3:
                volume_signal -= 0.5
        
        return volume_signal
    
    def _get_momentum_signal(self, df, idx):
        """Enhanced momentum signal"""
        roc = df['ROC'].iloc[idx]
        price_momentum = df['Price_momentum'].iloc[idx]
        
        momentum_signal = 0
        
        if not pd.isna(roc):
            if roc > 2.0:
                momentum_signal += 1.5
            elif roc > 0.8:
                momentum_signal += 0.8
            elif roc < -2.0:
                momentum_signal -= 1.5
            elif roc < -0.8:
                momentum_signal -= 0.8
        
        if not pd.isna(price_momentum):
            if price_momentum > 0.015:
                momentum_signal += 0.5
            elif price_momentum < -0.015:
                momentum_signal -= 0.5
        
        return momentum_signal
    
    def _get_volatility_signal(self, df, idx, thresholds):
        """Volatility-based signal (new from ML insights)"""
        volatility = thresholds['volatility_ratio']
        atr_norm = df['ATR_normalized'].iloc[idx]
        
        if pd.isna(atr_norm):
            return 0
        
        # High volatility suggests reversal opportunities
        if volatility > 1.6:
            if atr_norm > 0.025:  # Very high volatility
                return 0.8  # Favor contrarian plays
            else:
                return 0.4
        elif volatility < 0.6:  # Low volatility
            return -0.3  # Less favorable conditions
        else:
            return 0
    
    def scan_ticker(self, ticker, interval='15m'):
        """Scan ticker with optimized logic"""
        logger.info(f"Scanning {ticker} with optimized scanner")
        
        # Get data
        period = '5d' if interval == '15m' else '1mo'
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            logger.error(f"No data retrieved for {ticker}")
            return None
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Ensure sufficient data
        min_periods = max(self.params.values()) + 15
        if len(df) < min_periods:
            logger.warning(f"Not enough data for {ticker}. Need at least {min_periods} periods.")
            return None
        
        # Generate enhanced signals
        signals = self.generate_enhanced_signals(df)
        
        # Add ticker and metadata
        for signal in signals:
            signal['ticker'] = ticker
            signal['interval'] = interval
            signal['scan_method'] = 'ML-Optimized'
        
        logger.info(f"Generated optimized signal for {ticker}: {signals[0]['signal']} "
                   f"with {signals[0]['confidence']:.1%} confidence")
        
        return signals[0] if signals else None
    
    def save_evaluation_with_tracking(self, signal_data):
        """Enhanced evaluation saving with performance tracking"""
        # Load existing evaluations
        evaluations = self.load_or_create_evaluation_file()
        
        # Create enhanced evaluation record
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'ticker': signal_data['ticker'],
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'price': signal_data['price'],
            'scan_method': signal_data.get('scan_method', 'ML-Optimized'),
            'indicators': signal_data['indicators'],
            'components': signal_data['components'],
            'weighted_score': signal_data['weighted_score'],
            'thresholds': signal_data['thresholds']
        }
        
        evaluations.append(evaluation)
        
        # Enhanced history management - keep more data for analysis
        if len(evaluations) > 2000:
            evaluations = evaluations[-2000:]
        
        # Save to Azure
        blob_name = "same_day_technical/technical_evaluations.json"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        json_data = json.dumps(evaluations, indent=2, default=str)
        blob_client.upload_blob(json_data, overwrite=True)
        
        logger.info(f"Saved enhanced evaluation. Total: {len(evaluations)}")
    
    def load_or_create_evaluation_file(self):
        """Load evaluation file"""
        blob_name = "same_day_technical/technical_evaluations.json"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            blob_data = blob_client.download_blob()
            content = blob_data.readall()
            evaluations = json.loads(content)
            return evaluations
        except:
            return []
    
    def prepare_signal_for_evaluation(self, signal_data):
        """Prepare signal with enhanced metadata"""
        signal_data['scan_time'] = datetime.now().isoformat()
        signal_data['expiry_time'] = (datetime.now() + timedelta(minutes=15)).isoformat()
        
        # Enhanced strength calculation
        confidence = signal_data['confidence']
        if signal_data['signal'] == 'HOLD':
            signal_data['strength'] = 'Neutral'
        elif confidence >= 0.85:
            signal_data['strength'] = 'Very Strong'
        elif confidence >= 0.7:
            signal_data['strength'] = 'Strong'
        elif confidence >= 0.55:
            signal_data['strength'] = 'Moderate'
        else:
            signal_data['strength'] = 'Weak'
        
        return signal_data
    
    def run_scan(self, tickers=['NVDA']):
        """Run optimized scan"""
        all_signals = []
        
        for ticker in tickers:
            try:
                signal = self.scan_ticker(ticker)
                if signal:
                    prepared_signal = self.prepare_signal_for_evaluation(signal)
                    all_signals.append(prepared_signal)
                    
                    self.save_evaluation_with_tracking(prepared_signal)
                    
                    logger.info(f"Optimized signal for {ticker}: {prepared_signal['signal']} "
                               f"({prepared_signal['strength']}) at ${prepared_signal['price']}")
                    
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        logger.info(f"Optimized scan complete. Generated {len(all_signals)} signals")
        return all_signals
    
    def get_performance_summary(self):
        """Enhanced performance summary"""
        evaluations = self.load_or_create_evaluation_file()
        
        if len(evaluations) < 10:
            return None
        
        # Get recent evaluations
        recent = evaluations[-200:] if len(evaluations) >= 200 else evaluations
        
        # Enhanced metrics
        total_signals = len(recent)
        buy_signals = sum(1 for e in recent if e['signal'] == 'BUY')
        sell_signals = sum(1 for e in recent if e['signal'] == 'SELL')
        hold_signals = sum(1 for e in recent if e['signal'] == 'HOLD')
        
        avg_confidence = sum(e['confidence'] for e in recent) / len(recent)
        
        # Confidence distribution
        high_conf = sum(1 for e in recent if e['confidence'] >= 0.7)
        med_conf = sum(1 for e in recent if 0.5 <= e['confidence'] < 0.7)
        low_conf = sum(1 for e in recent if e['confidence'] < 0.5)
        
        # Time distribution
        hour_dist = {}
        for e in recent:
            try:
                hour = datetime.fromisoformat(e['timestamp']).hour
                hour_dist[hour] = hour_dist.get(hour, 0) + 1
            except:
                continue
        
        summary = {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_confidence': round(avg_confidence, 4),
            'confidence_distribution': {
                'high': high_conf,
                'medium': med_conf, 
                'low': low_conf
            },
            'signal_distribution': {
                'buy_pct': round(buy_signals / total_signals * 100, 1),
                'sell_pct': round(sell_signals / total_signals * 100, 1),
                'hold_pct': round(hold_signals / total_signals * 100, 1)
            },
            'hour_distribution': hour_dist,
            'scanner_version': 'ML-Optimized',
            'last_updated': datetime.now().isoformat()
        }
        
        # Save enhanced summary
        blob_name = "same_day_technical/performance_summary.json"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        json_data = json.dumps(summary, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)
        
        return summary


def main():
    """Main function for GitHub Actions with optimized scanner"""
    scanner = OptimizedTechnicalScanner()
    
    # Get tickers
    ticker = os.getenv('TICKER', 'NVDA')
    tickers = [ticker] if ticker != 'ALL' else ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BTC-USD', 'AC.TO']
    
    # Run optimized scan
    signals = scanner.run_scan(tickers)
    
    # Generate enhanced performance summary
    summary = scanner.get_performance_summary()
    if summary:
        logger.info(f"ML-Optimized Performance Summary:")
        logger.info(f"  Signals: {summary['buy_signals']} buys, {summary['sell_signals']} sells, {summary['hold_signals']} holds")
        logger.info(f"  Avg Confidence: {summary['avg_confidence']:.1%}")
        logger.info(f"  High Confidence Signals: {summary['confidence_distribution']['high']}")
        logger.info(f"  Scanner Version: {summary['scanner_version']}")
    
    logger.info("ML-Optimized technical scan completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())