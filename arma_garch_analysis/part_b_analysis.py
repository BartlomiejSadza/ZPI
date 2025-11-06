"""
ARMA-GARCH Analysis - Part B
Rolling window analysis with coverage tests
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_datareader.data as web
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RollingWindowAnalysis:
    """Rolling window ARMA-GARCH analysis"""

    def __init__(self, ticker='SPY', start_date='2017-01-01', end_date=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.returns = None
        self.results_df = None

    def download_data(self):
        """Download and prepare data"""
        print(f"Downloading {self.ticker} data...")

        try:
            # Try yfinance first
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(start=self.start_date, end=self.end_date)

            if len(data) > 0:
                self.returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
                print(f"Downloaded {len(self.returns)} observations")
                return self.returns
            else:
                raise ValueError("No data from yfinance")

        except Exception as e:
            print(f"yfinance failed: {e}")

            # Try arch library data
            try:
                from arch.data import sp500
                data = sp500.load()
                data.index = pd.to_datetime(data.index)

                # Filter to date range
                mask = (data.index >= self.start_date) & (data.index <= self.end_date)
                data = data[mask]

                if len(data) > 0:
                    self.returns = 100 * np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
                    print(f"Loaded {len(self.returns)} observations from arch library")
                    print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
                    return self.returns

            except Exception as e2:
                print(f"Arch library failed: {e2}")

            # Generate synthetic data as last resort
            print("Generating synthetic data...")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic S&P 500-like returns"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')

        np.random.seed(42)
        n = len(dates)

        # GARCH(1,1) with fat tails
        omega, alpha, beta = 0.01, 0.1, 0.85
        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(n):
            z = np.random.standard_t(df=5) / np.sqrt(5/3)
            returns[t] = 0.04 + np.sqrt(sigma2[t]) * z
            if t < n - 1:
                sigma2[t + 1] = omega + alpha * returns[t]**2 + beta * sigma2[t]

        self.returns = pd.Series(returns, index=dates, name='Returns')
        print(f"Generated {len(self.returns)} synthetic observations")
        print("Note: Using synthetic data with realistic GARCH properties")

        return self.returns

    def select_arma_order(self, data, max_p=5, max_q=5):
        """Select ARMA order using AIC"""
        best_aic = np.inf
        best_order = (0, 0)

        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(data, order=(p, 0, q))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, q)
                except:
                    continue

        return best_order

    def fit_and_forecast(self, train_data, p, q, garch_type='GARCH', dist='normal'):
        """
        Fit ARMA-GARCH model and forecast next return

        Parameters:
        -----------
        train_data : pd.Series
            Training data
        p, q : int
            ARMA orders
        garch_type : str
            'GARCH', 'EGARCH', or 'GJR-GARCH'
        dist : str
            Distribution: 'normal', 'studentst', or 't'

        Returns:
        --------
        dict with forecast mean, variance, CI, and diagnostics
        """
        try:
            # Model specification
            if garch_type == 'GARCH':
                vol_spec = {'vol': 'Garch', 'p': 1, 'q': 1, 'o': 0}
            elif garch_type == 'EGARCH':
                vol_spec = {'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 0}
            elif garch_type == 'GJR-GARCH':
                vol_spec = {'vol': 'Garch', 'p': 1, 'q': 1, 'o': 1}
            else:
                raise ValueError(f"Unknown GARCH type: {garch_type}")

            # Mean model
            if p > 0:
                mean_model = 'AR'
                lags = p
            else:
                mean_model = 'Zero'
                lags = None

            # Create and fit model
            if lags:
                model = arch_model(
                    train_data,
                    mean=mean_model,
                    lags=lags,
                    vol=vol_spec['vol'],
                    p=vol_spec['p'],
                    q=vol_spec['q'],
                    o=vol_spec['o'],
                    dist=dist
                )
            else:
                model = arch_model(
                    train_data,
                    mean=mean_model,
                    vol=vol_spec['vol'],
                    p=vol_spec['p'],
                    q=vol_spec['q'],
                    o=vol_spec['o'],
                    dist=dist
                )

            result = model.fit(disp='off', show_warning=False)

            # Forecast
            forecast = result.forecast(horizon=1, reindex=False)
            mean_forecast = forecast.mean.iloc[-1, 0]
            var_forecast = forecast.variance.iloc[-1, 0]

            # 95% Confidence interval
            z = stats.norm.ppf(0.975)
            ci_lower = mean_forecast - z * np.sqrt(var_forecast)
            ci_upper = mean_forecast + z * np.sqrt(var_forecast)

            # Standardized residuals for diagnostics
            std_resid = result.resid / result.conditional_volatility

            # Test if residuals follow assumed distribution
            if dist in ['normal']:
                # Jarque-Bera test for normality
                jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
                dist_test_pvalue = jb_pvalue
            else:
                # For t-distribution, use Kolmogorov-Smirnov
                ks_stat, ks_pvalue = stats.kstest(std_resid, 'norm')
                dist_test_pvalue = ks_pvalue

            return {
                'success': True,
                'mean': mean_forecast,
                'variance': var_forecast,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std_resid_mean': std_resid.mean(),
                'std_resid_std': std_resid.std(),
                'dist_test_pvalue': dist_test_pvalue,
                'aic': result.aic,
                'bic': result.bic
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mean': np.nan,
                'variance': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }

    def rolling_window_analysis(self, window_sizes=[100, 200, 500],
                                garch_models=['GARCH', 'EGARCH', 'GJR-GARCH'],
                                distributions=['normal', 'studentst', 't']):
        """
        Perform rolling window analysis

        Parameters:
        -----------
        window_sizes : list
            List of window sizes to test
        garch_models : list
            List of GARCH model types
        distributions : list
            List of distributions to test
        """
        print("\n" + "="*60)
        print("ROLLING WINDOW ANALYSIS - PART B")
        print("="*60)

        # Find first window end date (last observation of 2018)
        end_2018 = self.returns[self.returns.index.year == 2018].index[-1]
        print(f"\nFirst window ends: {end_2018.date()}")

        all_results = []

        # Iterate over window sizes
        for window_size in window_sizes:
            print(f"\n{'='*60}")
            print(f"Window Size: {window_size}")
            print(f"{'='*60}")

            # Find start index
            end_2018_idx = self.returns.index.get_loc(end_2018)
            start_idx = end_2018_idx - window_size + 1

            if start_idx < 0:
                print(f"Warning: Not enough data for window size {window_size}")
                continue

            # ARMA order selection for each year
            arma_orders = {}
            current_year = self.returns.index[start_idx].year
            start_date = self.returns.index[start_idx]

            for year in range(current_year, self.returns.index[-1].year + 1):
                # Get data up to beginning of year
                year_start = pd.Timestamp(f'{year}-01-01')
                if year_start > start_date:
                    year_data = self.returns.loc[start_date:year_start]
                    if len(year_data) >= 50:  # Minimum data requirement
                        order = self.select_arma_order(year_data.iloc[-min(window_size, len(year_data)):])
                        arma_orders[year] = order
                        print(f"  Year {year}: ARMA{order}")

            # If no orders selected, use default
            if not arma_orders:
                arma_orders[current_year] = (1, 1)
                print(f"  Using default ARMA(1,1)")

            # Rolling window forecast
            for i in tqdm(range(start_idx, len(self.returns) - 1),
                         desc=f"Window {window_size}"):

                # Training window
                train_data = self.returns.iloc[max(0, i - window_size + 1):i + 1]

                if len(train_data) < window_size:
                    continue

                # Actual next return
                actual_return = self.returns.iloc[i + 1]
                forecast_date = self.returns.index[i + 1]

                # Get ARMA order for current year
                current_year = self.returns.index[i].year
                if current_year in arma_orders:
                    p, q = arma_orders[current_year]
                else:
                    # Use order from previous year or default
                    prev_years = [y for y in arma_orders.keys() if y <= current_year]
                    if prev_years:
                        p, q = arma_orders[max(prev_years)]
                    else:
                        p, q = (1, 1)

                # Test each model combination
                for garch_model in garch_models:
                    for dist in distributions:
                        forecast_result = self.fit_and_forecast(
                            train_data, p, q, garch_model, dist
                        )

                        if forecast_result['success']:
                            # Check if actual return is in CI
                            in_ci = (forecast_result['ci_lower'] <= actual_return <=
                                   forecast_result['ci_upper'])

                            # Violation type
                            if not in_ci:
                                if actual_return < forecast_result['ci_lower']:
                                    violation_type = 'below'
                                else:
                                    violation_type = 'above'
                            else:
                                violation_type = 'none'

                            # Store results
                            all_results.append({
                                'date': forecast_date,
                                'window_size': window_size,
                                'garch_model': garch_model,
                                'distribution': dist,
                                'arma_p': p,
                                'arma_q': q,
                                'forecast_mean': forecast_result['mean'],
                                'forecast_var': forecast_result['variance'],
                                'ci_lower': forecast_result['ci_lower'],
                                'ci_upper': forecast_result['ci_upper'],
                                'actual_return': actual_return,
                                'in_ci': in_ci,
                                'violation_type': violation_type,
                                'dist_test_pvalue': forecast_result.get('dist_test_pvalue', np.nan),
                                'aic': forecast_result.get('aic', np.nan),
                                'bic': forecast_result.get('bic', np.nan)
                            })

        # Convert to DataFrame
        self.results_df = pd.DataFrame(all_results)

        # Save results
        self.results_df.to_csv('/home/user/ZPI/arma_garch_analysis/part_b_results.csv', index=False)
        print(f"\n\nResults saved to part_b_results.csv ({len(self.results_df)} records)")

        return self.results_df

    def analyze_coverage(self):
        """Analyze confidence interval coverage"""
        print("\n" + "="*60)
        print("CONFIDENCE INTERVAL COVERAGE ANALYSIS")
        print("="*60)

        # Overall coverage
        print("\nOverall Coverage Rates:")
        print("-" * 60)

        for window in self.results_df['window_size'].unique():
            window_data = self.results_df[self.results_df['window_size'] == window]
            coverage = window_data['in_ci'].mean() * 100

            print(f"\nWindow Size {window}:")
            print(f"  Coverage Rate: {coverage:.2f}%")
            print(f"  Violations:    {(~window_data['in_ci']).sum()}/{len(window_data)}")

            # By model
            print(f"\n  By GARCH Model:")
            for model in window_data['garch_model'].unique():
                model_data = window_data[window_data['garch_model'] == model]
                model_coverage = model_data['in_ci'].mean() * 100
                print(f"    {model:12s}: {model_coverage:6.2f}%")

            # By distribution
            print(f"\n  By Distribution:")
            for dist in window_data['distribution'].unique():
                dist_data = window_data[window_data['distribution'] == dist]
                dist_coverage = dist_data['in_ci'].mean() * 100
                print(f"    {dist:12s}: {dist_coverage:6.2f}%")

        # Violation analysis
        print("\n" + "="*60)
        print("VIOLATION ANALYSIS")
        print("="*60)

        violations = self.results_df[~self.results_df['in_ci']]

        if len(violations) > 0:
            print(f"\nTotal Violations: {len(violations)}")
            print(f"\nViolation Types:")
            print(violations['violation_type'].value_counts())

            print(f"\nViolations by Year:")
            violations_by_year = violations.groupby(violations['date'].dt.year).size()
            print(violations_by_year)

            # Check for clustering
            print(f"\nTemporal Clustering:")
            violations['date_only'] = violations['date'].dt.date
            consecutive_dates = violations['date_only'].diff().dt.days == 1
            if consecutive_dates.any():
                print(f"  Found {consecutive_dates.sum()} consecutive violation days")
            else:
                print(f"  No significant consecutive violations detected")

        # Statistical tests
        print("\n" + "="*60)
        print("STATISTICAL TESTS")
        print("="*60)

        # Test if coverage differs by model/distribution
        coverage_summary = self.results_df.groupby(
            ['window_size', 'garch_model', 'distribution']
        )['in_ci'].agg(['mean', 'count']).reset_index()

        coverage_summary.columns = ['window_size', 'garch_model', 'distribution',
                                    'coverage_rate', 'n_forecasts']
        coverage_summary = coverage_summary.sort_values('coverage_rate', ascending=False)

        print("\nTop 10 Model Combinations by Coverage:")
        print(coverage_summary.head(10).to_string(index=False))

        print("\nBottom 10 Model Combinations by Coverage:")
        print(coverage_summary.tail(10).to_string(index=False))

        return coverage_summary

    def plot_results(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        # 1. Coverage rates by window size and model
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Coverage by window size and GARCH model
        coverage_by_model = self.results_df.groupby(
            ['window_size', 'garch_model']
        )['in_ci'].mean().reset_index()

        for window in coverage_by_model['window_size'].unique():
            data = coverage_by_model[coverage_by_model['window_size'] == window]
            axes[0, 0].bar(
                [f"{m}\n{window}" for m in data['garch_model']],
                data['in_ci'] * 100,
                label=f'Window {window}',
                alpha=0.7
            )

        axes[0, 0].axhline(y=95, color='r', linestyle='--', label='Target 95%')
        axes[0, 0].set_ylabel('Coverage Rate (%)')
        axes[0, 0].set_title('Coverage Rate by GARCH Model and Window Size',
                            fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Coverage by distribution
        coverage_by_dist = self.results_df.groupby(
            ['window_size', 'distribution']
        )['in_ci'].mean().reset_index()

        window_sizes = sorted(coverage_by_dist['window_size'].unique())
        distributions = sorted(coverage_by_dist['distribution'].unique())
        x = np.arange(len(distributions))
        width = 0.25

        for i, window in enumerate(window_sizes):
            data = coverage_by_dist[coverage_by_dist['window_size'] == window]
            axes[0, 1].bar(
                x + i * width,
                data['in_ci'] * 100,
                width,
                label=f'Window {window}',
                alpha=0.7
            )

        axes[0, 1].axhline(y=95, color='r', linestyle='--', label='Target 95%')
        axes[0, 1].set_ylabel('Coverage Rate (%)')
        axes[0, 1].set_xlabel('Distribution')
        axes[0, 1].set_title('Coverage Rate by Distribution and Window Size',
                            fontweight='bold')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(distributions)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Violations over time
        violations = self.results_df[~self.results_df['in_ci']]

        for window in self.results_df['window_size'].unique():
            window_violations = violations[violations['window_size'] == window]
            if len(window_violations) > 0:
                violation_counts = window_violations.groupby(
                    window_violations['date'].dt.to_period('M')
                ).size()
                axes[1, 0].plot(
                    violation_counts.index.to_timestamp(),
                    violation_counts.values,
                    marker='o',
                    label=f'Window {window}',
                    alpha=0.7
                )

        axes[1, 0].set_ylabel('Number of Violations')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_title('CI Violations Over Time (Monthly)', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Violation types
        if len(violations) > 0:
            violation_summary = violations.groupby(
                ['window_size', 'violation_type']
            ).size().unstack(fill_value=0)

            violation_summary.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xlabel('Window Size')
            axes[1, 1].set_title('Violation Types by Window Size', fontweight='bold')
            axes[1, 1].legend(title='Violation Type')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig('/home/user/ZPI/arma_garch_analysis/part_b_coverage_analysis.png',
                   dpi=300, bbox_inches='tight')
        print("Saved: part_b_coverage_analysis.png")
        plt.close()

        # 2. Distribution test p-values over time
        fig, axes = plt.subplots(len(self.results_df['window_size'].unique()), 1,
                                figsize=(15, 12), sharex=True)

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for idx, window in enumerate(sorted(self.results_df['window_size'].unique())):
            window_data = self.results_df[self.results_df['window_size'] == window]

            for dist in window_data['distribution'].unique():
                dist_data = window_data[window_data['distribution'] == dist]
                # Average p-values across models for each date
                avg_pvalues = dist_data.groupby('date')['dist_test_pvalue'].mean()

                axes[idx].plot(avg_pvalues.index, avg_pvalues.values,
                             label=dist, alpha=0.7, linewidth=1)

            axes[idx].axhline(y=0.05, color='r', linestyle='--',
                            label='5% significance', linewidth=1)
            axes[idx].set_ylabel('p-value')
            axes[idx].set_title(f'Distribution Test p-values (Window {window})',
                              fontweight='bold')
            axes[idx].legend(loc='upper right', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 1])

        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.savefig('/home/user/ZPI/arma_garch_analysis/part_b_distribution_tests.png',
                   dpi=300, bbox_inches='tight')
        print("Saved: part_b_distribution_tests.png")
        plt.close()

        # 3. Forecast errors
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        for window in sorted(self.results_df['window_size'].unique()):
            window_data = self.results_df[
                (self.results_df['window_size'] == window) &
                (self.results_df['garch_model'] == 'GARCH') &
                (self.results_df['distribution'] == 'normal')
            ]

            forecast_errors = window_data['actual_return'] - window_data['forecast_mean']

            axes[0].plot(window_data['date'], forecast_errors,
                        label=f'Window {window}', alpha=0.6, linewidth=0.5)

        axes[0].axhline(y=0, color='k', linestyle='-', linewidth=1)
        axes[0].set_ylabel('Forecast Error (%)')
        axes[0].set_title('Forecast Errors Over Time (GARCH-Normal)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Histogram of forecast errors
        for window in sorted(self.results_df['window_size'].unique()):
            window_data = self.results_df[
                (self.results_df['window_size'] == window) &
                (self.results_df['garch_model'] == 'GARCH') &
                (self.results_df['distribution'] == 'normal')
            ]

            forecast_errors = window_data['actual_return'] - window_data['forecast_mean']

            axes[1].hist(forecast_errors, bins=50, alpha=0.5,
                        label=f'Window {window}', density=True)

        axes[1].set_xlabel('Forecast Error (%)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Forecast Errors', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/ZPI/arma_garch_analysis/part_b_forecast_errors.png',
                   dpi=300, bbox_inches='tight')
        print("Saved: part_b_forecast_errors.png")
        plt.close()

        print("\nAll visualizations completed!")


def main():
    """Main execution function for Part B"""
    print("="*60)
    print("ARMA-GARCH ANALYSIS - PART B")
    print("Rolling Window Analysis")
    print("="*60)

    # Initialize
    analysis = RollingWindowAnalysis(ticker='SPY', start_date='2017-01-01')

    # Download data
    analysis.download_data()

    # Rolling window analysis
    results = analysis.rolling_window_analysis(
        window_sizes=[100, 200, 500],
        garch_models=['GARCH', 'EGARCH', 'GJR-GARCH'],
        distributions=['normal', 'studentst', 't']
    )

    # Analyze coverage
    coverage_summary = analysis.analyze_coverage()

    # Plot results
    analysis.plot_results()

    print("\n" + "="*60)
    print("PART B ANALYSIS COMPLETED")
    print("="*60)
    print("\nResults saved to:")
    print("  - part_b_results.csv")
    print("  - part_b_coverage_analysis.png")
    print("  - part_b_distribution_tests.png")
    print("  - part_b_forecast_errors.png")


if __name__ == "__main__":
    main()
