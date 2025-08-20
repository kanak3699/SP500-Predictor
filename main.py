import pandas as pd
import yfinance as yf
from tqdm import tqdm
import warnings
import os
import numpy as np
import logging
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
yf.set_tz_cache_location("cache")

# --- Logger Configuration ---
def setup_logger():
    """Set up comprehensive logging for the S&P 500 prediction algorithm."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create logger
    logger = logging.getLogger('SP500_Predictor')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(funcName)20s:%(lineno)3d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler for detailed logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/sp500_prediction_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

# --- Enhanced Constants and Configuration ---
# S&P 500 Eligibility Criteria (updated for 2025)
MIN_MARKET_CAP_USD = 18.0 * 1_000_000_000  # $18.0 Billion
MIN_LIQUIDITY_RATIO = 0.75  # Annual dollar value traded / float-adjusted market cap
MIN_FREE_FLOAT = 0.50  # Minimum 50% of shares must be publicly available

# Enhanced Scoring Weights (rebalanced based on historical patterns)
# Addition Score Weights
W_MARKET_CAP = 0.25
W_LIQUIDITY = 0.20
W_SECTOR_BALANCE = 0.15
W_MOMENTUM = 0.10
W_PROFITABILITY = 0.15  # New: Quality metrics
W_STABILITY = 0.15      # New: Volatility and beta considerations

# Removal Score Weights
W_NEG_EARNINGS = 0.40
W_LOW_MARKET_CAP = 0.30
W_LOW_LIQUIDITY = 0.15
W_HIGH_VOLATILITY = 0.15  # New: Excessive volatility penalty

# --- Enhanced Phase 1: Data Collection ---

def get_sp500_constituents():
    """Enhanced S&P 500 scraping with comprehensive logging and validation."""
    logger.info("Starting S&P 500 constituents fetch from Wikipedia")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    try:
        logger.debug(f"Accessing URL: {url}")
        tables = pd.read_html(url)
        logger.debug(f"Found {len(tables)} tables on Wikipedia page")
        
        sp500_df = tables[0]
        logger.debug(f"Raw data shape: {sp500_df.shape}")
        logger.debug(f"Available columns: {list(sp500_df.columns)}")
        
        # Handle different possible column names
        column_mapping = {
            'Symbol': 'Ticker',
            'Ticker symbol': 'Ticker',
            'GICS Sector': 'GICS Sector',
            'Sector': 'GICS Sector',
            'Security': 'Security',
            'Company': 'Security'
        }
        
        mapped_columns = []
        for old_name, new_name in column_mapping.items():
            if old_name in sp500_df.columns:
                sp500_df.rename(columns={old_name: new_name}, inplace=True)
                mapped_columns.append(f"{old_name} -> {new_name}")
                
        logger.debug(f"Column mappings applied: {mapped_columns}")
        
        # Ensure we have required columns
        required_cols = ['Ticker', 'Security', 'GICS Sector']
        missing_cols = [col for col in required_cols if col not in sp500_df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns after mapping: {list(sp500_df.columns)}")
            return pd.DataFrame()
            
        sp500_df = sp500_df[required_cols]
        logger.debug(f"Data shape after column selection: {sp500_df.shape}")
        
        # Clean ticker symbols
        original_tickers = sp500_df['Ticker'].tolist()
        sp500_df['Ticker'] = sp500_df['Ticker'].str.replace('.', '-', regex=False)
        
        # Log any ticker symbol changes
        ticker_changes = []
        for orig, new in zip(original_tickers, sp500_df['Ticker'].tolist()):
            if orig != new:
                ticker_changes.append(f"{orig} -> {new}")
        
        if ticker_changes:
            logger.debug(f"Ticker symbol changes: {ticker_changes[:10]}{'...' if len(ticker_changes) > 10 else ''}")
        
        # Remove rows with missing tickers
        initial_count = len(sp500_df)
        sp500_df = sp500_df.dropna(subset=['Ticker'])
        final_count = len(sp500_df)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} rows with missing tickers")
        
        # Log sector distribution
        sector_counts = sp500_df['GICS Sector'].value_counts()
        logger.debug(f"Sector distribution: {dict(sector_counts.head(5))}")
        
        logger.info(f"Successfully fetched {len(sp500_df)} S&P 500 companies")
        return sp500_df
        
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {str(e)}", exc_info=True)
        return pd.DataFrame()

def get_russell1000_constituents(file_path='russell_1000_holdings.csv'):
    """Load Russell 1000 constituents with comprehensive logging and validation."""
    logger.info(f"Loading Russell 1000 constituents from: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Russell 1000 file not found: {file_path}")
        logger.error("Cannot proceed without authentic Russell 1000 data")
        logger.info("Please provide the Russell 1000 holdings CSV file")
        return pd.DataFrame()
        
    try:
        logger.debug(f"Reading CSV file: {file_path}")
        r1000_df = pd.read_csv(file_path)
        logger.debug(f"Raw data shape: {r1000_df.shape}")
        logger.debug(f"Available columns: {list(r1000_df.columns)}")
        
        # Handle different possible column names
        column_mapping = {
            'Symbol': 'Ticker',
            'Ticker Symbol': 'Ticker',
            'GICS Sector': 'GICS Sector',
            'Sector': 'GICS Sector',
            'Industry': 'GICS Sector'
        }
        
        mapped_columns = []
        for old_name, new_name in column_mapping.items():
            if old_name in r1000_df.columns:
                r1000_df.rename(columns={old_name: new_name}, inplace=True)
                mapped_columns.append(f"{old_name} -> {new_name}")
                
        logger.debug(f"Column mappings applied: {mapped_columns}")
        
        # Check for required columns
        required_cols = ['Ticker', 'GICS Sector']
        missing_cols = [col for col in required_cols if col not in r1000_df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in Russell 1000 file: {missing_cols}")
            logger.error(f"Available columns: {list(r1000_df.columns)}")
            return pd.DataFrame()
        
        # Keep only necessary columns
        r1000_df = r1000_df[required_cols]
        
        # Clean data
        initial_count = len(r1000_df)
        r1000_df.dropna(subset=['Ticker'], inplace=True)
        final_count = len(r1000_df)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} rows with missing tickers")
        
        # Clean ticker symbols
        r1000_df['Ticker'] = r1000_df['Ticker'].str.strip()
        r1000_df['GICS Sector'] = r1000_df['GICS Sector'].str.strip()
        
        # Log sector distribution
        sector_counts = r1000_df['GICS Sector'].value_counts()
        logger.debug(f"Sector distribution: {dict(sector_counts.head(5))}")
        
        logger.info(f"Successfully loaded {len(r1000_df)} Russell 1000 companies")
        return r1000_df
        
    except KeyError as e:
        logger.error(f"Column error in Russell 1000 file: {str(e)}")
        logger.error("Please ensure the CSV has 'Ticker' and 'Sector' columns")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to read Russell 1000 file: {str(e)}", exc_info=True)
        return pd.DataFrame()

def get_enhanced_stock_data(ticker):
    """Enhanced stock data collection with comprehensive logging."""
    logger.debug(f"Fetching data for ticker: {ticker}")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        logger.debug(f"Successfully retrieved info for {ticker}")
        
        # Basic metrics
        market_cap = info.get('marketCap', 0)
        last_price = info.get('previousClose', info.get('regularMarketPrice', 0))
        
        if market_cap == 0:
            logger.warning(f"{ticker}: Market cap not available")
        if last_price == 0:
            logger.warning(f"{ticker}: Last price not available")
        
        # Enhanced profitability metrics
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        profit_margins = info.get('profitMargins', 0)
        roe = info.get('returnOnEquity', 0)
        
        # More sophisticated profitability check
        is_profitable = (
            trailing_pe is not None and trailing_pe > 0 and
            profit_margins > 0
        )
        
        logger.debug(f"{ticker}: Profitable={is_profitable}, P/E={trailing_pe}, Margins={profit_margins}")
        
        # Enhanced liquidity metrics
        avg_volume_10day = info.get('averageDailyVolume10Day', 0)
        avg_volume_3month = info.get('averageVolume', avg_volume_10day)
        float_shares = info.get('floatShares', info.get('impliedSharesOutstanding', 0))
        
        # Calculate liquidity ratio with fallback
        volume_to_use = max(avg_volume_10day, avg_volume_3month)
        if float_shares > 0 and last_price > 0:
            annual_dollar_volume = (volume_to_use * last_price) * 252
            float_adjusted_market_cap = float_shares * last_price
            liquidity_ratio = annual_dollar_volume / float_adjusted_market_cap if float_adjusted_market_cap > 0 else 0
        else:
            liquidity_ratio = 0
            logger.warning(f"{ticker}: Could not calculate liquidity ratio (float_shares={float_shares}, price={last_price})")
            
        # Enhanced momentum and stability metrics
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', last_price)
        beta = info.get('beta', 1.0)
        
        # Calculate momentum
        if fifty_two_week_low and fifty_two_week_low > 0:
            momentum = (last_price - fifty_two_week_low) / fifty_two_week_low
        else:
            momentum = 0
            logger.warning(f"{ticker}: Could not calculate momentum (52w_low={fifty_two_week_low})")
            
        # Calculate stability score (inverse of volatility)
        if fifty_two_week_high and fifty_two_week_low and fifty_two_week_high > fifty_two_week_low:
            volatility = (fifty_two_week_high - fifty_two_week_low) / fifty_two_week_low
            stability_score = max(0, 1 - (volatility / 2))  # Normalize volatility
        else:
            stability_score = 0.5
            logger.warning(f"{ticker}: Using default stability score")
            
        # Free float calculation
        shares_outstanding = info.get('sharesOutstanding', float_shares)
        free_float_ratio = float_shares / shares_outstanding if shares_outstanding > 0 else 0.5
        
        # Get historical data for additional metrics
        annual_volatility = 0.5  # Default value
        try:
            logger.debug(f"Fetching 1-year historical data for {ticker}")
            hist = stock.history(period="1y")
            if not hist.empty:
                # Calculate additional volatility measure
                returns = hist['Close'].pct_change().dropna()
                if len(returns) > 30:
                    annual_volatility = returns.std() * np.sqrt(252)
                    stability_score = min(stability_score, max(0, 1 - annual_volatility))
                    logger.debug(f"{ticker}: Calculated annual volatility: {annual_volatility:.4f}")
                else:
                    logger.warning(f"{ticker}: Insufficient historical data for volatility calculation")
            else:
                logger.warning(f"{ticker}: No historical data available")
        except Exception as e:
            logger.warning(f"{ticker}: Error fetching historical data: {str(e)}")
        
        stock_data = {
            'Ticker': ticker,
            'Company Name': info.get('longName', ticker),
            'Market Cap': market_cap,
            'Is Profitable': is_profitable,
            'Profitability Score': min(1.0, max(0, (profit_margins * 10 + (roe or 0) * 2))),
            'Liquidity Ratio': liquidity_ratio,
            'Momentum': min(2.0, max(-1.0, momentum)),  # Cap momentum between -100% and 200%
            'Stability Score': stability_score,
            'Beta': beta or 1.0,
            'Free Float Ratio': free_float_ratio,
            'GICS Sector': info.get('sector', 'Unknown'),
            'Last Price': last_price,
            'Volume': volume_to_use,
            'PE Ratio': trailing_pe,
            'Profit Margins': profit_margins
        }
        
        logger.debug(f"{ticker}: Successfully processed all metrics")
        return stock_data
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        return None

# --- Enhanced Phase 2: Scoring Algorithms ---

def calculate_enhanced_addition_score(stock, sp500_sector_weights, broad_market_sector_weights):
    """Enhanced scoring algorithm with detailed logging."""
    logger.debug(f"Calculating addition score for {stock['Ticker']}")
    
    score = 0
    score_breakdown = {}
    
    # 1. Market Cap Score (with logarithmic scaling for very large caps)
    cap_ratio = stock['Market Cap'] / MIN_MARKET_CAP_USD
    if cap_ratio >= 1:
        cap_score = min(0.8 + 0.2 * np.log10(cap_ratio), 1.0)
    else:
        cap_score = cap_ratio
    
    score_component = W_MARKET_CAP * cap_score
    score += score_component
    score_breakdown['Market Cap'] = score_component
    
    # 2. Liquidity Score (enhanced with volume considerations)
    liq_score = min(stock['Liquidity Ratio'] / MIN_LIQUIDITY_RATIO, 1.0)
    score_component = W_LIQUIDITY * liq_score
    score += score_component
    score_breakdown['Liquidity'] = score_component
    
    # 3. Sector Balance Score (enhanced to consider sector momentum)
    sector = stock['GICS Sector']
    sp500_weight = sp500_sector_weights.get(sector, 0)
    broad_market_weight = broad_market_sector_weights.get(sector, 0)
    
    if broad_market_weight > sp500_weight and broad_market_weight > 0:
        sector_score = (broad_market_weight - sp500_weight) / broad_market_weight
    else:
        sector_score = 0
    
    score_component = W_SECTOR_BALANCE * sector_score
    score += score_component
    score_breakdown['Sector Balance'] = score_component
    
    # 4. Momentum Score (capped and normalized)
    momentum_score = (stock['Momentum'] + 1) / 3  # Normalize from [-1, 2] to [0, 1]
    momentum_score = max(0, min(1, momentum_score))
    score_component = W_MOMENTUM * momentum_score
    score += score_component
    score_breakdown['Momentum'] = score_component
    
    # 5. Profitability Score
    score_component = W_PROFITABILITY * stock['Profitability Score']
    score += score_component
    score_breakdown['Profitability'] = score_component
    
    # 6. Stability Score
    # Penalize extreme betas and reward stability
    beta_penalty = max(0, min(0.5, abs(stock['Beta'] - 1) / 2))
    adjusted_stability = max(0, stock['Stability Score'] - beta_penalty)
    score_component = W_STABILITY * adjusted_stability
    score += score_component
    score_breakdown['Stability'] = score_component
    
    final_score = round(score * 100, 2)
    
    logger.debug(f"{stock['Ticker']} addition score breakdown: {score_breakdown}")
    logger.debug(f"{stock['Ticker']} final addition score: {final_score}%")
    
    return final_score, score_breakdown

def calculate_enhanced_removal_score(stock):
    """Enhanced removal scoring with detailed logging."""
    logger.debug(f"Calculating removal score for {stock['Ticker']}")
    
    score = 0
    risk_factors = []
    score_breakdown = {}
    
    # 1. Profitability Failure (enhanced)
    if not stock['Is Profitable']:
        component_score = W_NEG_EARNINGS
        score += component_score
        score_breakdown['Unprofitable'] = component_score
        risk_factors.append("Unprofitable")
    elif stock['Profitability Score'] < 0.1:  # Very low profitability
        component_score = W_NEG_EARNINGS * 0.3
        score += component_score
        score_breakdown['Low Profitability'] = component_score
        risk_factors.append(f"Low profitability (Score: {stock['Profitability Score']:.2f})")
        
    # 2. Market Cap Deficit (with graduated penalties)
    if stock['Market Cap'] < MIN_MARKET_CAP_USD:
        deficit_ratio = (MIN_MARKET_CAP_USD - stock['Market Cap']) / MIN_MARKET_CAP_USD
        # More severe penalty for larger deficits
        deficit_score = min(1.0, deficit_ratio * 1.5)
        component_score = W_LOW_MARKET_CAP * deficit_score
        score += component_score
        score_breakdown['Market Cap Deficit'] = component_score
        risk_factors.append(f"Market cap below threshold (${stock['Market Cap']:,.0f})")
        
    # 3. Liquidity Issues
    if stock['Liquidity Ratio'] < MIN_LIQUIDITY_RATIO:
        liq_deficit = (MIN_LIQUIDITY_RATIO - stock['Liquidity Ratio']) / MIN_LIQUIDITY_RATIO
        component_score = W_LOW_LIQUIDITY * min(liq_deficit, 1.0)
        score += component_score
        score_breakdown['Low Liquidity'] = component_score
        risk_factors.append(f"Low liquidity (Ratio: {stock['Liquidity Ratio']:.2f})")
        
    # 4. High Volatility/Instability
    instability_score = 1 - stock['Stability Score']
    extreme_beta_penalty = max(0, (abs(stock['Beta']) - 2) / 3) if stock['Beta'] != 1.0 else 0
    volatility_risk = min(1.0, instability_score + extreme_beta_penalty)
    component_score = W_HIGH_VOLATILITY * volatility_risk
    score += component_score
    score_breakdown['High Volatility'] = component_score
    
    if volatility_risk > 0.5:
        risk_factors.append(f"High volatility (Beta: {stock['Beta']:.2f}, Stability: {stock['Stability Score']:.2f})")
    
    # 5. Free Float Issues
    if stock['Free Float Ratio'] < MIN_FREE_FLOAT:
        float_deficit = (MIN_FREE_FLOAT - stock['Free Float Ratio']) / MIN_FREE_FLOAT
        component_score = 0.1 * float_deficit  # Small additional penalty
        score += component_score
        score_breakdown['Low Free Float'] = component_score
        risk_factors.append(f"Low free float ({stock['Free Float Ratio']:.1%})")
        
    final_score = round(score * 100, 2)
    
    logger.debug(f"{stock['Ticker']} removal risk factors: {risk_factors}")
    logger.debug(f"{stock['Ticker']} final removal score: {final_score}%")
    
    return final_score, risk_factors, score_breakdown

# --- Markdown Report Generation ---

def generate_markdown_report(predictions, analysis_data, timestamp):
    """Generate comprehensive markdown report with predictions and reasoning."""
    report_filename = f"sp500_predictions_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"# S&P 500 Prediction Analysis Report\n\n")
        f.write(f"**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")
        
        # Executive Summary
        f.write("## Analysis Summary\n\n")
        
        if predictions['has_predictions']:
            f.write("### Predicted Changes\n\n")
            
            # Removals
            if predictions['removals']:
                f.write("#### Predicted Removals\n\n")
                for i, removal in enumerate(predictions['removals'][:5], 1):
                    f.write(f"**{i}. {removal['Ticker']} ({removal['Company Name']})**\n")
                    f.write(f"- **Sector:** {removal['GICS Sector']}\n")
                    f.write(f"- **Risk Score:** {removal['Risk Score']:.1f}%\n")
                    f.write(f"- **Market Cap:** ${removal['Market Cap']:,.0f}\n")
                    f.write(f"- **Primary Issues:** {', '.join(removal['Risk Factors'])}\n\n")
            
            # Additions
            if predictions['additions']:
                f.write("#### Predicted Additions\n\n")
                for i, addition in enumerate(predictions['additions'][:5], 1):
                    f.write(f"**{i}. {addition['Ticker']} ({addition['Company Name']})**\n")
                    f.write(f"- **Sector:** {addition['GICS Sector']}\n")
                    f.write(f"- **Fitness Score:** {addition['Fitness Score']:.1f}%\n")
                    f.write(f"- **Market Cap:** ${addition['Market Cap']:,.0f}\n")
                    f.write(f"- **Key Strengths:** Profitable: {addition['Is Profitable']}, ")
                    f.write(f"Liquidity Ratio: {addition['Liquidity Ratio']:.2f}\n\n")
            
            # Top Pairing
            if predictions['top_pairing']:
                pairing = predictions['top_pairing']
                f.write("###  Top Predicted Change\n\n")
                f.write(f"**Remove:** {pairing['removal']['Ticker']} ({pairing['removal']['Company Name']})\n")
                f.write(f"**Add:** {pairing['addition']['Ticker']} ({pairing['addition']['Company Name']})\n")
                f.write(f"**Reasoning:** {pairing['reasoning']}\n\n")
        else:
            f.write(" **No high-confidence predictions could be generated based on current criteria.**\n\n")
        
        # Detailed Analysis
        f.write("---\n\n## Detailed Analysis\n\n")
        
        # Market Overview
        f.write("### Market Overview\n\n")
        f.write(f"- **Current S&P 500 members analyzed:** {analysis_data['sp500_analyzed']}\n")
        f.write(f"- **Addition candidates analyzed:** {analysis_data['candidates_analyzed']}\n")
        f.write(f"- **Eligible addition candidates:** {analysis_data['eligible_candidates']}\n")
        f.write(f"- **Members with removal risk:** {analysis_data['high_risk_members']}\n\n")
        
        # Methodology
        f.write("### Methodology\n\n")
        f.write("#### Addition Scoring Criteria\n")
        f.write("- **Market Cap (25%):** Companies must exceed $18B threshold\n")
        f.write("- **Liquidity (20%):** Annual dollar volume vs float-adjusted market cap\n")
        f.write("- **Sector Balance (15%):** Alignment with broader market representation\n")
        f.write("- **Profitability (15%):** Earnings quality and profit margins\n")
        f.write("- **Stability (15%):** Volatility and beta considerations\n")
        f.write("- **Momentum (10%):** Recent price performance\n\n")
        
        f.write("#### Removal Risk Criteria\n")
        f.write("- **Unprofitable (40%):** Negative earnings or poor profitability\n")
        f.write("- **Market Cap Deficit (30%):** Below $18B threshold\n")
        f.write("- **Low Liquidity (15%):** Insufficient trading volume\n")
        f.write("- **High Volatility (15%):** Excessive price instability\n\n")
        
        # Detailed Candidate Lists
        if analysis_data.get('top_additions'):
            f.write("### Top Addition Candidates (Detailed)\n\n")
            f.write("| Rank | Ticker | Company | Sector | Market Cap | Fitness Score | Liquidity | Profitable |\n")
            f.write("|------|--------|---------|--------|------------|---------------|-----------|------------|\n")
            
            for i, candidate in enumerate(analysis_data['top_additions'][:15], 1):
                f.write(f"| {i} | {candidate['Ticker']} | {candidate['Company Name'][:25]}{'...' if len(candidate['Company Name']) > 25 else ''} | ")
                f.write(f"{candidate['GICS Sector'][:15]}{'...' if len(candidate['GICS Sector']) > 15 else ''} | ")
                f.write(f"${candidate['Market Cap']:,.0f} | {candidate['Fitness Score']:.1f}% | ")
                f.write(f"{candidate['Liquidity Ratio']:.2f} | {'✅' if candidate['Is Profitable'] else '❌'} |\n")
            f.write("\n")
        
        if analysis_data.get('top_removals'):
            f.write("### Top Removal Candidates (Detailed)\n\n")
            f.write("| Rank | Ticker | Company | Sector | Market Cap | Risk Score | Issues |\n")
            f.write("|------|--------|---------|--------|------------|------------|--------|\n")
            
            for i, candidate in enumerate(analysis_data['top_removals'][:15], 1):
                issues = ', '.join(candidate['Risk Factors'][:2])  # Show first 2 issues
                if len(candidate['Risk Factors']) > 2:
                    issues += '...'
                    
                f.write(f"| {i} | {candidate['Ticker']} | {candidate['Company Name'][:25]}{'...' if len(candidate['Company Name']) > 25 else ''} | ")
                f.write(f"{candidate['GICS Sector'][:15]}{'...' if len(candidate['GICS Sector']) > 15 else ''} | ")
                f.write(f"${candidate['Market Cap']:,.0f} | {candidate['Risk Score']:.1f}% | ")
                f.write(f"{issues[:30]}{'...' if len(issues) > 30 else ''} |\n")
            f.write("\n")
        
        # Sector Analysis
        if analysis_data.get('sector_analysis'):
            f.write("### 📈 Sector Analysis\n\n")
            f.write("#### Addition Candidates by Sector\n\n")
            sector_dist = analysis_data['sector_analysis'].get('addition_sectors', {})
            for sector, count in sorted(sector_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- **{sector}:** {count} candidates\n")
            f.write("\n")
        
        # Risk Analysis
        f.write("### ⚠️ Risk Assessment\n\n")
        f.write("#### Key Risks Identified\n")
        if analysis_data.get('top_removals'):
            risk_summary = {}
            for removal in analysis_data['top_removals'][:10]:
                for risk in removal['Risk Factors']:
                    if risk not in risk_summary:
                        risk_summary[risk] = 0
                    risk_summary[risk] += 1
            
            for risk, count in sorted(risk_summary.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{risk}:** {count} companies affected\n")
        f.write("\n")
        
        # Disclaimer
        f.write("---\n\n")
        f.write("## ⚠️ Important Disclaimer\n\n")
        f.write("This analysis is based purely on quantitative financial metrics and historical S&P 500 ")
        f.write("inclusion criteria. Actual S&P 500 index committee decisions involve additional ")
        f.write("qualitative factors including:\n\n")
        f.write("- Company financial viability and outlook\n")
        f.write("- Sector representation balance\n")
        f.write("- Corporate governance standards\n")
        f.write("- Market disruption considerations\n")
        f.write("- Committee discretion and timing\n\n")
        f.write("**This report is for educational and analytical purposes only and should not be used ")
        f.write("for investment decisions.**\n\n")
        
        # Technical Details
        f.write("---\n\n")
        f.write("## 🔧 Technical Details\n\n")
        f.write(f"- **Analysis Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Data Sources:** Yahoo Finance, Wikipedia, Russell 1000 Holdings\n")
        f.write(f"- **Market Cap Threshold:** ${MIN_MARKET_CAP_USD:,.0f}\n")
        f.write(f"- **Liquidity Threshold:** {MIN_LIQUIDITY_RATIO}\n")
        f.write(f"- **Free Float Threshold:** {MIN_FREE_FLOAT:.0%}\n")
        f.write(f"- **Algorithm Version:** Enhanced Multi-Factor Model v2.0\n\n")
    
    logger.info(f"Markdown report generated: {report_filename}")
    return report_filename

# --- Enhanced Phase 3: Main Execution ---

def analyze_sector_trends(df):
    """Analyze sector representation and trends with logging."""
    logger.debug(f"Analyzing sector trends for {len(df)} companies")
    
    sector_stats = df.groupby('GICS Sector').agg({
        'Market Cap': ['count', 'sum', 'mean'],
        'Momentum': 'mean',
        'Stability Score': 'mean'
    }).round(2)
    
    sector_stats.columns = ['Count', 'Total_Cap', 'Avg_Cap', 'Avg_Momentum', 'Avg_Stability']
    sector_stats = sector_stats.sort_values('Total_Cap', ascending=False)
    
    logger.debug(f"Sector analysis completed. Top sector by market cap: {sector_stats.index[0]}")
    return sector_stats

def main():
    """Enhanced main execution with comprehensive logging and markdown report generation."""
    timestamp = datetime.now()
    logger.info("="*60)
    logger.info("STARTING ENHANCED S&P 500 PREDICTION ALGORITHM")
    logger.info("="*60)
    logger.info(f"Analysis Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Minimum Market Cap Threshold: ${MIN_MARKET_CAP_USD:,}")
    logger.info(f"Minimum Liquidity Ratio: {MIN_LIQUIDITY_RATIO}")
    logger.info(f"Minimum Free Float: {MIN_FREE_FLOAT:.0%}")
    
    # Initialize prediction results structure
    predictions = {
        'has_predictions': False,
        'additions': [],
        'removals': [],
        'top_pairing': None
    }
    
    analysis_data = {
        'sp500_analyzed': 0,
        'candidates_analyzed': 0,
        'eligible_candidates': 0,
        'high_risk_members': 0,
        'failed_data_fetches': 0
    }
    
    # 1. Get company universes - AUTHENTIC DATA ONLY
    logger.info("Phase 1: Fetching authentic company data")
    sp500_df = get_sp500_constituents()
    r1000_df = get_russell1000_constituents()
    
    if sp500_df.empty:
        logger.error("CRITICAL: Could not fetch S&P 500 data. Cannot proceed.")
        print("\nERROR: Could not fetch S&P 500 data. Please check your internet connection.")
        return
        
    if r1000_df.empty:
        logger.error("CRITICAL: Could not load Russell 1000 data. Cannot proceed without candidate universe.")
        print("\nERROR: Could not load Russell 1000 data from CSV file.")
        print("Please ensure 'russell_1000_holdings.csv' exists and contains 'Ticker' and 'Sector' columns.")
        return
    
    sp500_tickers = set(sp500_df['Ticker'])
    r1000_tickers = set(r1000_df['Ticker'])
    
    logger.info(f"S&P 500 companies loaded: {len(sp500_tickers)}")
    logger.info(f"Russell 1000 companies loaded: {len(r1000_tickers)}")
    
    # Create candidate universe (Russell 1000 - S&P 500)
    candidate_tickers = list(r1000_tickers - sp500_tickers)
    overlap_count = len(r1000_tickers & sp500_tickers)
    
    logger.info(f"Candidate universe size: {len(candidate_tickers)} (Russell 1000 - S&P 500)")
    logger.info(f"Overlap between indices: {overlap_count} companies")
    
    if len(candidate_tickers) == 0:
        logger.error("No candidates found. Russell 1000 and S&P 500 have identical tickers.")
        print("\nERROR: No addition candidates found. Check data sources.")
        return
    
    # 2. Enhanced data collection with detailed progress tracking
    logger.info("Phase 2: Collecting enhanced financial data")
    all_tickers = list(sp500_tickers) + candidate_tickers
    logger.info(f"Total companies to process: {len(all_tickers)}")
    
    all_stock_data = []
    failed_tickers = []
    successful_count = 0
    
    print(f"\nFetching enhanced financial data for {len(all_tickers)} companies...")
    
    for i, ticker in enumerate(tqdm(all_tickers, desc="Processing Stocks")):
        data = get_enhanced_stock_data(ticker)
        if data:
            all_stock_data.append(data)
            successful_count += 1
        else:
            failed_tickers.append(ticker)
            
        # Log progress every 50 stocks
        if (i + 1) % 50 == 0:
            logger.debug(f"Progress: {i+1}/{len(all_tickers)} processed, {successful_count} successful")
    
    analysis_data['failed_data_fetches'] = len(failed_tickers)
    
    logger.info(f"Data collection completed: {successful_count} successful, {len(failed_tickers)} failed")
    
    if failed_tickers:
        logger.warning(f"Failed to fetch data for {len(failed_tickers)} tickers")
        logger.debug(f"Failed tickers: {failed_tickers[:20]}{'...' if len(failed_tickers) > 20 else ''}")
        print(f"\nWarning: Could not fetch data for {len(failed_tickers)} companies")
    
    if not all_stock_data:
        logger.error("CRITICAL: No stock data collected. Cannot proceed.")
        print("\nERROR: No stock data was successfully collected. Check internet connection and data sources.")
        return
    
    all_stocks_df = pd.DataFrame(all_stock_data)
    logger.info(f"Created dataset with {len(all_stocks_df)} companies and {len(all_stocks_df.columns)} metrics")
    
    # Separate datasets
    current_sp500_data = all_stocks_df[all_stocks_df['Ticker'].isin(sp500_tickers)].copy()
    candidate_data = all_stocks_df[all_stocks_df['Ticker'].isin(candidate_tickers)].copy()
    
    analysis_data['sp500_analyzed'] = len(current_sp500_data)
    analysis_data['candidates_analyzed'] = len(candidate_data)
    
    logger.info(f"Dataset separation completed:")
    logger.info(f"- Current S&P 500 members with data: {len(current_sp500_data)}")
    logger.info(f"- Addition candidates with data: {len(candidate_data)}")
    
    print(f"\nDataset Summary:")
    print(f"- Current S&P 500 members analyzed: {len(current_sp500_data)}")
    print(f"- Addition candidates analyzed: {len(candidate_data)}")
    
    # 3. Enhanced candidate analysis
    logger.info("Phase 3: Analyzing addition candidates")
    print(f"\n{'='*50}")
    print("ADDITION CANDIDATES ANALYSIS")
    print('='*50)
    
    # Apply enhanced eligibility criteria
    eligible_candidates = candidate_data[
        (candidate_data['Market Cap'] >= MIN_MARKET_CAP_USD) &
        (candidate_data['Is Profitable'] == True) &
        (candidate_data['Liquidity Ratio'] >= MIN_LIQUIDITY_RATIO) &
        (candidate_data['Free Float Ratio'] >= MIN_FREE_FLOAT)
    ].copy()
    
    analysis_data['eligible_candidates'] = len(eligible_candidates)
    
    logger.info(f"Eligibility screening results:")
    logger.info(f"- Market cap >= ${MIN_MARKET_CAP_USD:,}: {len(candidate_data[candidate_data['Market Cap'] >= MIN_MARKET_CAP_USD])}")
    logger.info(f"- Profitable: {len(candidate_data[candidate_data['Is Profitable'] == True])}")
    logger.info(f"- Liquidity >= {MIN_LIQUIDITY_RATIO}: {len(candidate_data[candidate_data['Liquidity Ratio'] >= MIN_LIQUIDITY_RATIO])}")
    logger.info(f"- Free float >= {MIN_FREE_FLOAT:.0%}: {len(candidate_data[candidate_data['Free Float Ratio'] >= MIN_FREE_FLOAT])}")
    logger.info(f"- Meeting ALL criteria: {len(eligible_candidates)}")
    
    print(f"Candidates meeting basic eligibility: {len(eligible_candidates)}/{len(candidate_data)}")
    
    if not eligible_candidates.empty:
        # Calculate sector weights for scoring
        logger.info("Calculating sector weights for scoring")
        sp500_sector_weights = current_sp500_data['GICS Sector'].value_counts(normalize=True)
        broad_market_sector_weights = all_stocks_df['GICS Sector'].value_counts(normalize=True)
        
        logger.debug(f"S&P 500 sector weights: {dict(sp500_sector_weights.head(3))}")
        logger.debug(f"Broad market sector weights: {dict(broad_market_sector_weights.head(3))}")
        
        # Calculate enhanced scores with detailed breakdown
        logger.info("Calculating fitness scores for eligible candidates")
        scores_and_breakdowns = eligible_candidates.apply(
            lambda row: calculate_enhanced_addition_score(row, sp500_sector_weights, broad_market_sector_weights),
            axis=1
        )
        
        eligible_candidates['Fitness Score'] = [score for score, _ in scores_and_breakdowns]
        eligible_candidates['Score Breakdown'] = [breakdown for _, breakdown in scores_and_breakdowns]
        
        # Get top candidates
        top_additions = eligible_candidates.sort_values(by='Fitness Score', ascending=False).head(20)
        analysis_data['top_additions'] = top_additions.to_dict('records')
        
        # Store for predictions
        predictions['additions'] = top_additions.head(10).to_dict('records')
        
        logger.info(f"Top addition candidate: {top_additions.iloc[0]['Ticker']} (Score: {top_additions.iloc[0]['Fitness Score']}%)")
        
        print(f"\nTop {min(15, len(top_additions))} Addition Candidates:")
        display_cols = ['Ticker', 'Company Name', 'GICS Sector', 'Market Cap', 'Liquidity Ratio', 'Fitness Score']
        display_df = top_additions[display_cols].copy()
        display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}")
        print(display_df.to_string(index=False, max_colwidth=25))
        
        # Sector analysis of candidates
        sector_dist = top_additions.head(10)['GICS Sector'].value_counts()
        analysis_data['sector_analysis'] = {'addition_sectors': dict(sector_dist)}
        logger.info(f"Top 10 candidates by sector: {dict(sector_dist)}")
        print(f"\nSector Distribution of Top 10 Candidates:")
        for sector, count in sector_dist.items():
            print(f"  {sector}: {count}")

    # 4. Enhanced removal analysis  
    logger.info("Phase 4: Analyzing removal candidates")
    print(f"\n{'='*50}")
    print("REMOVAL CANDIDATES ANALYSIS")
    print('='*50)
    
    logger.info("Calculating removal risk scores for current S&P 500 members")
    
    # Calculate removal scores with detailed breakdown
    removal_scores_and_details = current_sp500_data.apply(calculate_enhanced_removal_score, axis=1)
    current_sp500_data['Risk Score'] = [score for score, _, _ in removal_scores_and_details]
    current_sp500_data['Risk Factors'] = [factors for _, factors, _ in removal_scores_and_details]
    current_sp500_data['Risk Breakdown'] = [breakdown for _, _, breakdown in removal_scores_and_details]
    
    # Use a threshold for significant risk
    risk_threshold = 5.0  # 5% risk score threshold
    potential_removals = current_sp500_data[current_sp500_data['Risk Score'] > risk_threshold]
    
    analysis_data['high_risk_members'] = len(potential_removals)
    
    logger.info(f"Risk analysis results:")
    logger.info(f"- Companies above {risk_threshold}% risk threshold: {len(potential_removals)}")
    logger.info(f"- Average risk score: {current_sp500_data['Risk Score'].mean():.2f}%")
    logger.info(f"- Maximum risk score: {current_sp500_data['Risk Score'].max():.2f}%")
    
    if not potential_removals.empty:
        top_removals = potential_removals.sort_values(by='Risk Score', ascending=False).head(20)
        analysis_data['top_removals'] = top_removals.to_dict('records')
        
        # Store for predictions
        predictions['removals'] = top_removals.head(10).to_dict('records')
        
        logger.info(f"Highest removal risk: {top_removals.iloc[0]['Ticker']} (Risk: {top_removals.iloc[0]['Risk Score']}%)")
        
        print(f"Current members with removal risk: {len(potential_removals)}")
        print(f"\nTop {min(15, len(top_removals))} Removal Candidates:")
        display_cols = ['Ticker', 'Company Name', 'GICS Sector', 'Market Cap', 'Risk Score']
        display_df = top_removals[display_cols].copy()
        display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}")
        print(display_df.to_string(index=False, max_colwidth=25))
    else:
        # Show lowest performers anyway
        bottom_performers = current_sp500_data.sort_values(by='Risk Score', ascending=False).head(15)
        analysis_data['top_removals'] = bottom_performers.to_dict('records')
        
        logger.info("No current members exceed risk threshold")
        print("No current S&P 500 members flagged with significant removal risk.")
        print(f"\nLowest Performing Current Members (by Risk Score):")
        display_cols = ['Ticker', 'Company Name', 'GICS Sector', 'Market Cap', 'Risk Score']
        display_df = bottom_performers[display_cols].copy()
        display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}")
        print(display_df.to_string(index=False, max_colwidth=25))

    # 5. Enhanced final prediction with comprehensive reasoning
    logger.info("Phase 5: Generating final predictions")
    print(f"\n{'='*50}")
    print("FINAL PREDICTIONS")
    print('='*50)
    
    # Check if we have both removal and addition candidates
    has_removal_candidates = predictions['removals']
    has_addition_candidates = predictions['additions']
    
    if has_removal_candidates and has_addition_candidates:
        predictions['has_predictions'] = True
        
        primary_removal = predictions['removals'][0]
        removal_sector = primary_removal['GICS Sector']
        
        logger.info(f"Primary removal candidate: {primary_removal['Ticker']} from {removal_sector} sector")
        
        # Enhanced sector-based matching
        sector_matches = [add for add in predictions['additions'] if add['GICS Sector'] == removal_sector]
        
        if sector_matches:
            primary_addition = sector_matches[0]
            reasoning = f"Sector balance strategy: Replacing {primary_removal['Ticker']} with {primary_addition['Ticker']} maintains sector diversity in '{removal_sector}'"
            logger.info(f"Sector balance strategy: Found replacement in same sector ({removal_sector})")
            print(f"✓ Sector Balance Strategy Applied")
            print(f"  Maintaining sector diversity in '{removal_sector}'")
        else:
            primary_addition = predictions['additions'][0]
            reasoning = f"Best available candidate strategy: {primary_addition['Ticker']} offers strongest fundamentals despite sector shift from {removal_sector} to {primary_addition['GICS Sector']}"
            logger.info(f"Best candidate strategy: No suitable same-sector replacement")
            print(f"✓ Best Available Candidate Strategy")
            print(f"  No strong candidates in removal sector '{removal_sector}'")
            
        predictions['top_pairing'] = {
            'removal': primary_removal,
            'addition': primary_addition,
            'reasoning': reasoning
        }
            
        logger.info(f"Final prediction: Remove {primary_removal['Ticker']}, Add {primary_addition['Ticker']}")
            
        print(f"\n PREDICTED REMOVAL:")
        print(f"   {primary_removal['Ticker']} ({primary_removal['Company Name']})")
        print(f"   Sector: {primary_removal['GICS Sector']}")
        print(f"   Risk Score: {primary_removal['Risk Score']}%")
        print(f"   Market Cap: ${primary_removal['Market Cap']:,.0f}")
        print(f"   Key Issues: {', '.join(primary_removal['Risk Factors'][:3])}")
        
        print(f"\n PREDICTED ADDITION:")
        print(f"   {primary_addition['Ticker']} ({primary_addition['Company Name']})")
        print(f"   Sector: {primary_addition['GICS Sector']}")
        print(f"   Fitness Score: {primary_addition['Fitness Score']}%")
        print(f"   Market Cap: ${primary_addition['Market Cap']:,.0f}")
        print(f"   Strengths: {'Profitable' if primary_addition['Is Profitable'] else 'Unprofitable'}, High Liquidity ({primary_addition['Liquidity Ratio']:.2f})")
        
    else:
        logger.warning("Could not generate high-confidence predictions")
        print("Could not generate high-confidence predictions")
        print("Insufficient candidates meeting criteria or risk thresholds")
        
        # Show best available info
        if has_addition_candidates:
            best_addition = predictions['additions'][0]
            logger.info(f"Best available addition candidate: {best_addition['Ticker']}")
            print(f"\n   Best Addition Candidate: {best_addition['Ticker']} (Score: {best_addition['Fitness Score']}%)")
            
        if analysis_data['sp500_analyzed'] > 0:
            highest_risk = analysis_data['top_removals'][0]
            logger.info(f"Highest risk current member: {highest_risk['Ticker']}")
            print(f"   Highest Risk Current Member: {highest_risk['Ticker']} (Risk: {highest_risk['Risk Score']}%)")
    
    # 6. Generate comprehensive markdown report
    logger.info("Phase 6: Generating markdown report")
    report_filename = generate_markdown_report(predictions, analysis_data, timestamp)
    
    # 7. Summary output
    print(f"\n{'='*50}")
    print("ANALYSIS SUMMARY")
    print('='*50)
    
    end_time = datetime.now()
    logger.info(f"Analysis completed successfully at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"Analysis completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"S&P 500 members analyzed: {analysis_data['sp500_analyzed']}")
    print(f"Addition candidates analyzed: {analysis_data['candidates_analyzed']}")
    print(f"Eligible addition candidates: {analysis_data['eligible_candidates']}")
    print(f"Members with removal risk (>{risk_threshold}%): {analysis_data['high_risk_members']}")
    print(f"Data fetch failures: {analysis_data['failed_data_fetches']}")
    
    print(f"\n**COMPREHENSIVE REPORT GENERATED: {report_filename}**")
    print(f"\n Note: Predictions are based on quantitative analysis of financial metrics")
    print(f"   and historical S&P 500 criteria. Actual decisions involve additional")
    print(f"   qualitative factors and committee discretion.")
    print(f"\nDetailed logs saved to: logs/sp500_prediction_{timestamp.strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.info("Enhanced S&P 500 prediction algorithm completed successfully")
    logger.info(f"Markdown report saved as: {report_filename}")
    logger.info("="*60)

if __name__ == '__main__':
    main()
