"""
ARMA-GARCH Analysis - Part A
Analysis of S&P 500 daily returns with ARMA-GARCH models
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_datareader.data as web
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ARMAGARCHAnalysis:
    """Class for ARMA-GARCH analysis of financial time series"""

    def __init__(self, ticker='SPY', start_date='2017-01-01', end_date=None):
        """
        Initialize the analysis

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (default: SPY for S&P 500 ETF)
        start_date : str
            Start date for data download
        end_date : str or None
            End date for data download (None = today)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.best_arma_order = None
        self.garch_results = {}

    def download_data(self):
        """Download price data and calculate returns"""
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")

        try:
            # Try downloading with yfinance
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(start=self.start_date, end=self.end_date)

            if len(data) == 0:
                raise ValueError("No data downloaded")

            # Calculate log returns
            self.data = data['Close']
            self.returns = 100 * np.log(self.data / self.data.shift(1)).dropna()

            print(f"Downloaded {len(self.returns)} daily returns")
            print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")

        except Exception as e:
            print(f"Error downloading from Yahoo Finance: {e}")
            print("Attempting alternative download method...")

            # Alternative: Try with different parameters
            import time
            time.sleep(2)

            try:
                data = yf.download(self.ticker, start=self.start_date, end=self.end_date,
                                 progress=False, threads=False)

                if len(data) == 0:
                    raise ValueError("No data downloaded with alternative method")

                self.data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                self.returns = 100 * np.log(self.data / self.data.shift(1)).dropna()

                print(f"Downloaded {len(self.returns)} daily returns")
                print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")

            except Exception as e2:
                print(f"yfinance download failed: {e2}")

                # Try pandas_datareader as last resort
                if DATAREADER_AVAILABLE:
                    print("Trying pandas_datareader...")
                    try:
                        data = web.DataReader(self.ticker, 'stooq', self.start_date, self.end_date)
                        data = data.sort_index()  # Stooq returns data in reverse order

                        if len(data) == 0:
                            raise ValueError("No data from pandas_datareader")

                        self.data = data['Close']
                        self.returns = 100 * np.log(self.data / self.data.shift(1)).dropna()

                        print(f"Downloaded {len(self.returns)} daily returns using pandas_datareader")
                        print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")

                    except Exception as e3:
                        print(f"pandas_datareader also failed: {e3}")
                        print("\nAll external sources blocked. Using arch library sample data...")
                        return self._use_sample_data()
                else:
                    print("\nAll external sources blocked. Using arch library sample data...")
                    return self._use_sample_data()

        return self.returns

    def _use_sample_data(self):
        """Use sample data from arch library or generate synthetic data"""
        try:
            # Try to use arch library's built-in SP500 data
            from arch.data import sp500
            data = sp500.load()

            # Filter to our date range
            data.index = pd.to_datetime(data.index)
            mask = (data.index >= self.start_date) & (data.index <= self.end_date)
            data = data[mask]

            if len(data) > 0:
                self.data = data['Adj Close']
                self.returns = 100 * np.log(self.data / self.data.shift(1)).dropna()

                print(f"Loaded {len(self.returns)} daily returns from arch library")
                print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")

                return self.returns

        except Exception as e:
            print(f"Could not load arch sample data: {e}")

        # Generate synthetic S&P 500-like data
        print("Generating synthetic S&P 500-like data...")

        # Create date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')

        # Generate returns with realistic properties:
        # - Mean: ~0.04% daily (~10% annually)
        # - Volatility: ~1% daily (~16% annually)
        # - Fat tails (t-distribution with df=5)
        # - Volatility clustering (GARCH effect)

        np.random.seed(42)
        n = len(dates)

        # GARCH(1,1) simulation
        omega = 0.01
        alpha = 0.1
        beta = 0.85

        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance

        for t in range(n):
            # Generate standardized residual from t-distribution (fat tails)
            z = np.random.standard_t(df=5) / np.sqrt(5/3)  # Normalize to unit variance

            # Calculate return
            returns[t] = 0.04 + np.sqrt(sigma2[t]) * z

            # Update variance for next period
            if t < n - 1:
                sigma2[t + 1] = omega + alpha * returns[t]**2 + beta * sigma2[t]

        self.returns = pd.Series(returns, index=dates, name='Returns')
        self.data = pd.Series(100 * np.exp(np.cumsum(returns/100)), index=dates, name='Price')

        print(f"Generated {len(self.returns)} synthetic daily returns")
        print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        print("Note: Using synthetic data with S&P 500-like properties (GARCH, fat tails)")

        return self.returns

    def descriptive_statistics(self):
        """Calculate and display descriptive statistics"""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)

        stats_dict = {
            'Mean': self.returns.mean(),
            'Std Dev': self.returns.std(),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis(),
            'Min': self.returns.min(),
            'Max': self.returns.max(),
            'Jarque-Bera': stats.jarque_bera(self.returns)[0],
            'JB p-value': stats.jarque_bera(self.returns)[1]
        }

        for key, value in stats_dict.items():
            print(f"{key:20s}: {value:12.6f}")

        return stats_dict

    def test_stationarity(self):
        """Perform Augmented Dickey-Fuller test"""
        print("\n" + "="*60)
        print("STATIONARITY TEST (Augmented Dickey-Fuller)")
        print("="*60)

        adf_result = adfuller(self.returns, autolag='AIC')

        print(f"ADF Statistic:        {adf_result[0]:.6f}")
        print(f"p-value:              {adf_result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in adf_result[4].items():
            print(f"  {key:5s}:             {value:.6f}")

        if adf_result[1] < 0.05:
            print("\nConclusion: Series is stationary (p < 0.05)")
        else:
            print("\nConclusion: Series is NOT stationary (p >= 0.05)")

        return adf_result

    def plot_diagnostics(self, filename='part_a_diagnostics.png'):
        """Plot diagnostic charts for returns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series plot
        axes[0, 0].plot(self.returns.index, self.returns, linewidth=0.5)
        axes[0, 0].set_title(f'{self.ticker} Daily Returns (%)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Returns (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # Distribution
        axes[0, 1].hist(self.returns, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Returns', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Returns (%)')
        axes[0, 1].set_ylabel('Density')

        # Normal distribution overlay
        mu, sigma = self.returns.mean(), self.returns.std()
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # ACF
        plot_acf(self.returns, lags=40, ax=axes[1, 0])
        axes[1, 0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # PACF
        plot_pacf(self.returns, lags=40, ax=axes[1, 1])
        axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'/home/user/ZPI/arma_garch_analysis/{filename}', dpi=300, bbox_inches='tight')
        print(f"\nDiagnostic plot saved as {filename}")
        plt.close()

    def test_arch_effect(self):
        """Test for ARCH effects in residuals"""
        print("\n" + "="*60)
        print("ARCH EFFECT TEST (Ljung-Box on Squared Returns)")
        print("="*60)

        # Ljung-Box test on squared returns
        lb_result = acorr_ljungbox(self.returns**2, lags=[10, 20, 30], return_df=True)
        print(lb_result)

        if (lb_result['lb_pvalue'] < 0.05).any():
            print("\nConclusion: ARCH effects are present (at least one p-value < 0.05)")
        else:
            print("\nConclusion: No significant ARCH effects detected")

        return lb_result

    def select_arma_order(self, max_p=5, max_q=5):
        """Select optimal ARMA order using AIC"""
        print("\n" + "="*60)
        print("ARMA MODEL SELECTION (using AIC)")
        print("="*60)

        best_aic = np.inf
        best_order = None

        results_table = []

        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue

                try:
                    model = ARIMA(self.returns, order=(p, 0, q))
                    fitted = model.fit()
                    aic = fitted.aic
                    results_table.append({'p': p, 'q': q, 'AIC': aic})

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, q)

                except Exception as e:
                    continue

        # Display top 10 models
        results_df = pd.DataFrame(results_table).sort_values('AIC').head(10)
        print("\nTop 10 ARMA models by AIC:")
        print(results_df.to_string(index=False))

        self.best_arma_order = best_order
        print(f"\nSelected ARMA order: ({best_order[0]}, {best_order[1]})")
        print(f"AIC: {best_aic:.4f}")

        return best_order

    def fit_garch_models(self, distributions=['normal', 'studentst', 't']):
        """
        Fit GARCH, EGARCH, and GJR-GARCH models with different distributions

        Parameters:
        -----------
        distributions : list
            List of distributions to test: 'normal', 'studentst', 't', 'skewt', 'ged'
        """
        print("\n" + "="*60)
        print("FITTING GARCH MODELS")
        print("="*60)

        p, q = self.best_arma_order

        # Model specifications
        garch_specs = {
            'GARCH': {'vol': 'Garch', 'p': 1, 'q': 1},
            'EGARCH': {'vol': 'EGARCH', 'p': 1, 'q': 1},
            'GJR-GARCH': {'vol': 'Garch', 'p': 1, 'o': 1, 'q': 1}
        }

        for model_name, spec in garch_specs.items():
            print(f"\n{model_name} Models:")
            print("-" * 60)

            for dist in distributions:
                try:
                    # Create mean model
                    if p > 0 and q > 0:
                        mean_model = 'ARX'
                        lags = p
                    elif p > 0:
                        mean_model = 'AR'
                        lags = p
                    else:
                        mean_model = 'Zero'
                        lags = None

                    # Fit model
                    if lags is not None:
                        model = arch_model(
                            self.returns,
                            mean=mean_model,
                            lags=lags,
                            vol=spec['vol'],
                            p=spec['p'],
                            q=spec['q'],
                            o=spec.get('o', 0),
                            dist=dist
                        )
                    else:
                        model = arch_model(
                            self.returns,
                            mean=mean_model,
                            vol=spec['vol'],
                            p=spec['p'],
                            q=spec['q'],
                            o=spec.get('o', 0),
                            dist=dist
                        )

                    result = model.fit(disp='off', show_warning=False)

                    # Store results
                    key = f"{model_name}_{dist}"
                    self.garch_results[key] = {
                        'model': model,
                        'result': result,
                        'aic': result.aic,
                        'bic': result.bic,
                        'log_likelihood': result.loglikelihood
                    }

                    print(f"  {dist:12s}: AIC={result.aic:10.2f}, BIC={result.bic:10.2f}, LogLik={result.loglikelihood:10.2f}")

                except Exception as e:
                    print(f"  {dist:12s}: Failed - {str(e)[:50]}")

        # Find best model
        best_model = min(self.garch_results.items(), key=lambda x: x[1]['aic'])
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model[0]} (AIC = {best_model[1]['aic']:.2f})")
        print(f"{'='*60}")

        return self.garch_results

    def model_diagnostics(self, model_key, filename_prefix='part_a'):
        """Perform diagnostic tests on fitted GARCH model"""
        print(f"\n{'='*60}")
        print(f"DIAGNOSTICS FOR {model_key}")
        print(f"{'='*60}")

        result = self.garch_results[model_key]['result']

        # Print summary
        print(result.summary())

        # Standardized residuals
        std_resid = result.resid / result.conditional_volatility

        # Normality tests
        print("\n" + "="*60)
        print("STANDARDIZED RESIDUALS TESTS")
        print("="*60)

        jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
        print(f"Jarque-Bera test:     {jb_stat:.4f} (p-value: {jb_pvalue:.4f})")

        ks_stat, ks_pvalue = stats.kstest(std_resid, 'norm')
        print(f"Kolmogorov-Smirnov:   {ks_stat:.4f} (p-value: {ks_pvalue:.4f})")

        # Ljung-Box on standardized residuals
        lb_resid = acorr_ljungbox(std_resid, lags=[10, 20, 30], return_df=True)
        print("\nLjung-Box test on standardized residuals:")
        print(lb_resid)

        # Ljung-Box on squared standardized residuals
        lb_resid2 = acorr_ljungbox(std_resid**2, lags=[10, 20, 30], return_df=True)
        print("\nLjung-Box test on squared standardized residuals:")
        print(lb_resid2)

        # Plot diagnostics
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Standardized residuals
        axes[0, 0].plot(std_resid.index, std_resid, linewidth=0.5)
        axes[0, 0].set_title('Standardized Residuals', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Std. Residuals')
        axes[0, 0].grid(True, alpha=0.3)

        # Conditional volatility
        axes[0, 1].plot(result.conditional_volatility.index, result.conditional_volatility, linewidth=0.8)
        axes[0, 1].set_title('Conditional Volatility', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # ACF of standardized residuals
        plot_acf(std_resid, lags=40, ax=axes[1, 0])
        axes[1, 0].set_title('ACF of Standardized Residuals', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # ACF of squared standardized residuals
        plot_acf(std_resid**2, lags=40, ax=axes[1, 1])
        axes[1, 1].set_title('ACF of Squared Standardized Residuals', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # Histogram with normal overlay
        axes[2, 0].hist(std_resid, bins=50, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(std_resid.min(), std_resid.max(), 100)
        axes[2, 0].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Standard Normal')
        axes[2, 0].set_title('Distribution of Standardized Residuals', fontsize=11, fontweight='bold')
        axes[2, 0].set_xlabel('Standardized Residuals')
        axes[2, 0].set_ylabel('Density')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(std_resid, dist="norm", plot=axes[2, 1])
        axes[2, 1].set_title('Q-Q Plot', fontsize=11, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'{filename_prefix}_model_diagnostics_{model_key}.png'
        plt.savefig(f'/home/user/ZPI/arma_garch_analysis/{filename}', dpi=300, bbox_inches='tight')
        print(f"\nDiagnostic plots saved as {filename}")
        plt.close()

        return std_resid

    def forecast_next_return(self, model_key, alpha=0.05):
        """
        Forecast next return with confidence interval

        Parameters:
        -----------
        model_key : str
            Key of the model to use for forecasting
        alpha : float
            Significance level for confidence interval (default: 0.05 for 95% CI)
        """
        print(f"\n{'='*60}")
        print(f"FORECASTING WITH {model_key}")
        print(f"{'='*60}")

        result = self.garch_results[model_key]['result']

        # Make forecast
        forecast = result.forecast(horizon=1, reindex=False)

        # Extract forecast values
        mean_forecast = forecast.mean.iloc[-1, 0]
        variance_forecast = forecast.variance.iloc[-1, 0]

        # Confidence interval
        z_alpha = stats.norm.ppf(1 - alpha/2)
        ci_lower = mean_forecast - z_alpha * np.sqrt(variance_forecast)
        ci_upper = mean_forecast + z_alpha * np.sqrt(variance_forecast)

        print(f"\nForecast for next return:")
        print(f"  Mean:           {mean_forecast:8.4f}%")
        print(f"  Volatility:     {np.sqrt(variance_forecast):8.4f}%")
        print(f"  95% CI:         [{ci_lower:8.4f}%, {ci_upper:8.4f}%]")

        # Get actual next return if available
        # Note: This will only work if we have held out data
        print(f"\nNote: Actual next return verification requires out-of-sample data")

        return {
            'mean': mean_forecast,
            'variance': variance_forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }


def main():
    """Main execution function for Part A"""
    print("="*60)
    print("ARMA-GARCH ANALYSIS - PART A")
    print("S&P 500 Daily Returns (2017-01-01 to Present)")
    print("="*60)

    # Initialize analysis
    analysis = ARMAGARCHAnalysis(ticker='SPY', start_date='2017-01-01')

    # Download data
    analysis.download_data()

    # Descriptive statistics
    analysis.descriptive_statistics()

    # Stationarity test
    analysis.test_stationarity()

    # Plot diagnostics
    analysis.plot_diagnostics()

    # Test for ARCH effects
    analysis.test_arch_effect()

    # Select ARMA order
    analysis.select_arma_order(max_p=5, max_q=5)

    # Fit GARCH models
    analysis.fit_garch_models(distributions=['normal', 'studentst', 't'])

    # Get best model
    best_model_key = min(analysis.garch_results.items(), key=lambda x: x[1]['aic'])[0]

    # Diagnostics for best model
    analysis.model_diagnostics(best_model_key)

    # Forecast next return
    forecast = analysis.forecast_next_return(best_model_key)

    # Save results summary
    results_summary = {
        'ticker': analysis.ticker,
        'start_date': analysis.start_date,
        'end_date': analysis.end_date,
        'n_observations': len(analysis.returns),
        'arma_order': analysis.best_arma_order,
        'best_model': best_model_key,
        'forecast': forecast
    }

    # Save to file
    import json
    with open('/home/user/ZPI/arma_garch_analysis/part_a_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        results_json = {
            k: (v.tolist() if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, (np.integer, np.floating)) else v)
            for k, v in results_summary.items()
        }
        json.dump(results_json, f, indent=2, default=str)

    print("\n" + "="*60)
    print("PART A ANALYSIS COMPLETED")
    print("="*60)
    print("\nResults saved to:")
    print("  - part_a_results.json")
    print("  - part_a_diagnostics.png")
    print(f"  - part_a_model_diagnostics_{best_model_key}.png")


if __name__ == "__main__":
    main()
