# S&P 500 Index Change Predictor

A quantitative analysis tool that predicts potential additions and removals from the S&P 500 index using real-time financial data and historical inclusion criteria.

## Core Concept

The S&P 500 index committee regularly reviews and adjusts the index composition based on specific financial and market criteria. This tool automates that analysis by:

1. **Fetching real-time data** for current S&P 500 members and Russell 1000 candidates
2. **Applying S&P 500 eligibility criteria** (market cap, liquidity, profitability, free float)
3. **Scoring companies** using a multi-factor algorithm based on historical patterns
4. **Predicting changes** by identifying at-risk current members and strong candidates
5. **Generating comprehensive reports** with detailed reasoning for each prediction

## How to Run

### 1. Clone the Repository 
-  https://github.com/kanak3699/SP500-Predictor.git

### 2. Install Python Dependencies  

- Run the following command in terminal to install dependencies 
```
pip install pandas yfinance tqdm numpy
```

### 3. Run the app
- Type the following command in terminal 
```
python main.py
```

## Algorithm Overview

### Addition Scoring (Weighted Multi-Factor Model)
- **Market Cap (25%)**: Companies must exceed $18B threshold with logarithmic scaling
- **Liquidity (20%)**: Annual dollar volume vs float-adjusted market cap ratio
- **Sector Balance (15%)**: Alignment with broader market sector representation
- **Profitability (15%)**: Earnings quality, profit margins, and ROE metrics
- **Stability (15%)**: Price volatility and beta considerations
- **Momentum (10%)**: Recent 52-week price performance

### Removal Risk Assessment
- **Unprofitable (40%)**: Negative earnings or deteriorating profitability
- **Market Cap Deficit (30%)**: Below $18B threshold with graduated penalties
- **Low Liquidity (15%)**: Insufficient trading volume relative to market cap
- **High Volatility (15%)**: Excessive price instability or extreme beta values

## Detailed Calculation Methods

### Addition Score Components

#### 1. Market Cap Score (25% weight)
- **Base calculation**: `market_cap / $18B_threshold`
- **Logarithmic scaling**: For caps > $18B, applies `0.8 + 0.2 * log10(ratio)` to prevent mega-caps from dominating
- **Example**: $50B company gets score of ~0.94, not 2.78, creating more balanced competition

#### 2. Liquidity Score (20% weight)
- **Formula**: `(daily_volume × price × 252_trading_days) / (float_shares × price)`
- **Threshold**: Must exceed 0.75 ratio (75% of float traded annually)
- **Data sources**: 10-day and 3-month average volumes, uses higher value for stability

#### 3. Sector Balance Score (15% weight)
- **Calculation**: `(broad_market_sector_weight - sp500_sector_weight) / broad_market_sector_weight`
- **Purpose**: Rewards candidates from underrepresented sectors
- **Example**: If Technology is 25% of Russell 1000 but only 20% of S&P 500, Tech candidates get bonus

#### 4. Profitability Score (15% weight)
- **Base requirement**: `trailing_PE > 0 AND profit_margins > 0`
- **Score formula**: `min(1.0, profit_margins × 10 + ROE × 2)`
- **Quality metrics**: Combines current profitability with return efficiency

#### 5. Momentum Score (10% weight)
- **Formula**: `(current_price - 52_week_low) / 52_week_low`
- **Normalization**: Converts to 0-1 scale: `(momentum + 1) / 3`
- **Range**: Caps between -100% and +200% to prevent extreme outliers

#### 6. Stability Score (15% weight)
- **Base calculation**: `1 - (52_week_range / 52_week_low)`
- **Beta adjustment**: Penalizes extreme betas: `max(0, score - |beta - 1| / 2)`
- **Volatility overlay**: Uses annualized return volatility when available: `returns_std × √252`

### Removal Risk Components

#### 1. Negative Earnings Assessment (40% weight)
- **Primary check**: `trailing_PE ≤ 0 OR profit_margins ≤ 0`
- **Secondary check**: Very low profitability score < 0.1 gets 30% penalty
- **Data validation**: Confirms with multiple profitability indicators (PE, margins, ROE)
- **Severity scaling**: Recent losses weighted more heavily than historical patterns

#### 2. Market Cap Deficit (30% weight)
- **Threshold**: Companies below $18B face graduated penalties
- **Calculation**: `(threshold - actual_cap) / threshold × 1.5`
- **Severity scaling**: 50% below threshold = maximum penalty, smaller deficits get proportional penalties
- **Trend analysis**: Considers sustained decline vs temporary dip

#### 3. Low Liquidity Risk (15% weight)
- **Calculation**: `(0.75 - actual_ratio) / 0.75` when below threshold
- **Impact factors**: 
  - Float shares availability
  - Daily trading volume consistency
  - Market maker presence (indirectly through volume patterns)
- **Warning signs**: Declining volume trends over time

#### 4. High Volatility Penalty (15% weight)
- **Instability score**: `1 - stability_score`
- **Extreme beta penalty**: Additional penalty for |beta| > 2.0
- **Combined risk**: `min(1.0, instability + extreme_beta_penalty)`
- **Market context**: Adjusts for overall market volatility periods

### Advanced Calculation Features

#### Free Float Analysis
- **Calculation**: `float_shares / total_outstanding_shares`
- **Minimum threshold**: 50% must be publicly tradeable
- **Impact**: Affects liquidity calculations and market accessibility
- **Corporate structure consideration**: Identifies family/founder control issues

#### Sector Weight Calculations
- **S&P 500 weights**: Current index composition by GICS sector
- **Broad market weights**: Russell 1000 sector distribution as market proxy
- **Dynamic adjustment**: Updates with each analysis run for current market conditions
- **Balance scoring**: Rewards additions that improve sector representation alignment

#### Data Validation & Quality Checks
- **Multiple data sources**: Cross-validates key metrics across different API endpoints
- **Historical consistency**: Flags sudden metric changes for manual review
- **Error handling**: Graceful degradation when specific metrics unavailable
- **Logging transparency**: Documents all calculation steps and data sources used

### Key Thresholds (2025 Criteria)
- **Minimum Market Cap**: $18.0 billion
- **Minimum Liquidity Ratio**: 0.75
- **Minimum Free Float**: 50%
- **Risk Score Threshold**: 5.0%

## Output Files

### 1. Markdown Report (`sp500_predictions_YYYYMMDD_HHMMSS.md`)
- Executive summary with top predictions
- Detailed candidate tables with rankings
- Sector analysis and risk assessment
- Methodology and technical details

### 2. Log File (`logs/sp500_prediction_YYYYMMDD_HHMMSS.log`)
- Detailed execution logs
- Data fetching progress
- Score calculations and breakdowns
- Error handling and warnings

## Architecture

### Core Components
1. **Data Collection**: Multi-source data aggregation with error handling
2. **Eligibility Screening**: S&P 500 criteria validation
3. **Scoring Engine**: Multi-factor weighted scoring algorithms
4. **Prediction Logic**: Sector-balanced matching with fallback strategies
5. **Report Generation**: Structured markdown output with detailed reasoning

### Algorithm Features
- **Sector Balance Priority**: Attempts to maintain sector diversity
- **Logarithmic Market Cap Scaling**: Handles mega-cap companies appropriately
- **Volatility Adjustment**: Penalizes excessive price instability
- **Liquidity Normalization**: Accounts for float-adjusted market cap

## Important Disclaimers

- **Quantitative Analysis Only**: Based purely on financial metrics and historical criteria
- **Committee Discretion**: Actual S&P 500 decisions involve qualitative factors
- **Educational Purpose**: Not intended for investment decisions
- **Data Limitations**: Subject to data availability and API limitations

##  Example Prediction Output

```
   PREDICTED REMOVAL:
   XYZ (XYZ Company Inc.)
   Sector: Information Technology  
   Risk Score: 25.3%
   Key Issues: Market cap below threshold, Low liquidity

   PREDICTED ADDITION:
   ABC (ABC Corporation)
   Sector: Information Technology
   Fitness Score: 87.5%
   Strengths: Profitable, High Liquidity (1.25)
```

## Troubleshooting

### Common Issues
- **Missing Russell 1000 file**: Ensure `russell_1000_holdings.csv` exists
- **API rate limits**: Yahoo Finance may throttle requests during high usage
- **Network connectivity**: Required for real-time data fetching
- **Column naming**: CSV file must have `Ticker` and `Sector` columns

### Performance Notes
- **Processing time**: ~5-10 minutes for full analysis
- **Memory usage**: ~200-500MB depending on data size
- **Network requests**: 500-1000 API calls to Yahoo Finance

##  Accuracy & Validation

The algorithm uses historical S&P 500 inclusion patterns and official criteria. While predictions are based on quantitative analysis, actual index committee decisions incorporate additional qualitative factors including corporate governance, market disruption considerations, and strategic sector balance decisions.

---

**Last Updated**: August 08, 2025  
**Python Compatibility**: 3.7+
