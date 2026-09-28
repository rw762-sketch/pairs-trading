"""
Main orchestration script for pairs trading strategy.
Runs the complete pipeline from data fetching to analysis.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pairs_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import all modules
from data_fetcher import fetch_price_data, clean_price_data, split_data, get_sp500_tickers
from cointegration import find_cointegrated_pairs, rank_pairs, results_to_dataframe
from signal_generation import generate_all_signals
from backtester import backtest_all_pairs, combine_portfolio_results
from visualization import (
    plot_equity_curve, plot_spread_and_zscore, plot_cointegration_heatmap,
    plot_drawdown, generate_comprehensive_report
)
from parameter_optimization import ParameterOptimizer, compare_in_sample_vs_out_of_sample
from reporting import create_run_directory, report_path, update_report_index


class PairsTradingPipeline:
    """Main pipeline for pairs trading strategy."""
    
    def __init__(self, config=None):
        """
        Initialize the pipeline.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = dict(config or self._get_default_config())
        self._output_base_dir = Path(self.config['output_dir'])
        self.run_dir = None
        self.run_started_at = None
        self.requested_stock_count = None
        self.downloaded_stock_count = None
        self.price_data = None
        self.train_data = None
        self.test_data = None
        self.cointegrated_pairs = None
        self.signals = None
        self.backtest_results = None
        self.portfolio_results = None
    
    @staticmethod
    def _get_default_config():
        """Get default configuration."""
        return {
            # Data parameters
            'start_date': (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'num_stocks': 50,  # Number of stocks to analyze
            'min_history_days': 252,  # Minimum trading days
            'train_test_split': 0.7,  # 70% train, 30% test
            
            # Cointegration parameters
            'coint_threshold': 0.05,  # P-value threshold
            'adf_threshold': 0.05,
            
            # Trading parameters
            'entry_zscore': 2.0,
            'exit_zscore': 0.0,
            'transaction_cost': 0.001,  # 0.1%
            'initial_capital': 100000,
            
            # Output parameters
            'output_dir': 'results',
            'run_name': 'pairs-trading-backtest',
            'save_plots': True,
            'generate_report': True
        }
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        self.run_started_at = datetime.now(ZoneInfo('America/New_York'))
        self.run_dir = None
        self.config['output_dir'] = str(self._output_base_dir)
        logger.info("Starting pairs trading pipeline")
        
        # Step 1: Fetch data
        logger.info("Step 1: Fetching data...")
        self._fetch_data()
        
        # Step 2: Clean data
        logger.info("Step 2: Cleaning data...")
        self._clean_data()
        self._prepare_output_directory()
        
        # Step 3: Find cointegrated pairs
        logger.info("Step 3: Finding cointegrated pairs...")
        self._find_pairs()
        
        # Step 4: Generate signals
        logger.info("Step 4: Generating trading signals...")
        self._generate_signals()
        
        # Step 5: Run backtests
        logger.info("Step 5: Running backtests...")
        self._backtest_strategy()
        
        # Step 6: Analyze and visualize
        logger.info("Step 6: Generating analysis and visualizations...")
        self._analyze_results()
        update_report_index()
        
        logger.info("Pipeline complete!")
    
    def _fetch_data(self):
        """Fetch price data."""
        tickers = get_sp500_tickers()
        self.requested_stock_count = len(tickers)
        
        logger.info(f"Fetching data for {len(tickers)} stocks")
        
        self.price_data = fetch_price_data(
            tickers,
            self.config['start_date'],
            self.config['end_date']
        )
        self.downloaded_stock_count = self.price_data.shape[1]
        
        logger.info(f"Fetched data shape: {self.price_data.shape}")
    
    def _clean_data(self):
        """Clean and split data."""
        self.price_data = clean_price_data(
            self.price_data,
            min_history_days=self.config['min_history_days']
        )
        
        self.train_data, self.test_data = split_data(
            self.price_data,
            train_end_ratio=self.config['train_test_split']
        )
        
        logger.info(f"Data after cleaning: {self.price_data.shape}")

    def _prepare_output_directory(self):
        """Give each run its own dated folder and record the actual data window."""
        if self.run_started_at is None:
            self.run_started_at = datetime.now(ZoneInfo('America/New_York'))
        self.run_dir = create_run_directory(
            self._output_base_dir,
            self.config.get('run_name', 'pairs-trading-backtest'),
            self.price_data.shape[1],
            started_at=self.run_started_at,
        )
        self.config['output_dir'] = str(self.run_dir)

        def data_window(data):
            if data is None or data.empty:
                return {'first_date': None, 'last_date': None, 'rows': 0}
            return {
                'first_date': pd.Timestamp(data.index.min()).date().isoformat(),
                'last_date': pd.Timestamp(data.index.max()).date().isoformat(),
                'rows': len(data),
            }

        metadata = {
            'run_name': self.config.get('run_name', 'pairs-trading-backtest'),
            'started_at': self.run_started_at.isoformat(),
            'timezone': 'America/New_York',
            'configured_start_date': self.config['start_date'],
            'configured_end_date': self.config['end_date'],
            'requested_stock_count': self.requested_stock_count,
            'downloaded_stock_count': self.downloaded_stock_count,
            'retained_stock_count': self.price_data.shape[1],
            'folder_stock_count_basis': 'retained after cleaning',
            'stocks': self.price_data.columns.tolist(),
            'all_data': data_window(self.price_data),
            'training_data': data_window(self.train_data),
            'test_data': data_window(self.test_data),
        }
        report_path(self.run_dir, 'run-metadata', 'json').write_text(
            json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
        )
        logger.info(f"Run outputs: {self.run_dir}")
    
    def _find_pairs(self):
        """Find cointegrated pairs."""
        self.cointegrated_pairs = find_cointegrated_pairs(
            self.train_data,
            coint_threshold=self.config['coint_threshold']
        )
        
        logger.info(f"Found {len(self.cointegrated_pairs)} cointegrated pairs")
        
        # Save cointegration results
        os.makedirs(self.config['output_dir'], exist_ok=True)
        pairs_df = results_to_dataframe(self.cointegrated_pairs)
        pairs_df.to_csv(report_path(self.config['output_dir'], 'cointegrated-pairs', 'csv'))
        
        # Create heatmap
        if self.config['save_plots']:
            fig = plot_cointegration_heatmap(self.cointegrated_pairs)
            if fig:
                fig.savefig(
                    report_path(self.config['output_dir'], 'cointegration-heatmap', 'png'),
                    dpi=150, bbox_inches='tight'
                )
    
    def _generate_signals(self):
        """Generate trading signals."""
        self.signals = generate_all_signals(
            self.test_data,
            self.cointegrated_pairs,
            entry_threshold=self.config['entry_zscore'],
            exit_threshold=self.config['exit_zscore']
        )
        
        logger.info(f"Generated signals for {len(self.signals)} pairs")
    
    def _backtest_strategy(self):
        """Run backtests."""
        self.backtest_results = backtest_all_pairs(
            self.signals,
            initial_capital=self.config['initial_capital'],
            transaction_cost=self.config['transaction_cost']
        )
        
        self.portfolio_results = combine_portfolio_results(self.backtest_results)
        
        if self.portfolio_results:
            logger.info(f"Portfolio Sharpe Ratio: {self.portfolio_results['metrics']['Sharpe_Ratio']:.2f}")
            logger.info(f"Portfolio Return: {self.portfolio_results['metrics']['Total_Return']:.2%}")
    
    def _analyze_results(self):
        """Analyze results and generate visualizations."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        if not self.backtest_results:
            logger.warning("No backtest results to analyze")
            return
        
        # Plot results for first few pairs
        pair_ids = list(self.backtest_results.keys())[:3]
        
        for pair_id in pair_ids:
            result = self.backtest_results[pair_id]
            
            if self.config['save_plots']:
                # Equity curve
                fig = plot_equity_curve(
                    result['equity_curve'],
                    title=f"{pair_id} - Equity Curve"
                )
                fig.savefig(
                    report_path(self.config['output_dir'], f'{pair_id}-equity', 'png'),
                    dpi=150, bbox_inches='tight'
                )
                
                # Drawdown
                fig = plot_drawdown(result['equity_curve'])
                fig.savefig(
                    report_path(self.config['output_dir'], f'{pair_id}-drawdown', 'png'),
                    dpi=150, bbox_inches='tight'
                )
        
        # Plot portfolio equity curve
        if self.portfolio_results and self.config['save_plots']:
            fig = plot_equity_curve(
                self.portfolio_results['equity_curve'],
                title="Portfolio Equity Curve (Equal Weighted)"
            )
            fig.savefig(
                report_path(self.config['output_dir'], 'portfolio-equity', 'png'),
                dpi=150, bbox_inches='tight'
            )
        
        # Generate report
        if self.config['generate_report']:
            generate_comprehensive_report(
                self.backtest_results,
                self.portfolio_results,
                output_path=report_path(self.config['output_dir'], 'backtest-report', 'txt')
            )
        
        # Save detailed results
        self._save_results()
    
    def _save_results(self):
        """Save detailed results to CSV."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Summary metrics
        summary = []
        for pair_id, result in self.backtest_results.items():
            metrics = result['metrics'].copy()
            metrics['Pair'] = pair_id
            summary.append(metrics)
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(
            report_path(self.config['output_dir'], 'backtest-summary', 'csv'),
            index=False
        )
        
        # Portfolio metrics
        if self.portfolio_results:
            portfolio_metrics = pd.DataFrame([self.portfolio_results['metrics']])
            portfolio_metrics.to_csv(
                report_path(self.config['output_dir'], 'portfolio-metrics', 'csv'),
                index=False
            )
        
        logger.info(f"Results saved to {self.config['output_dir']}")


def main():
    """Main entry point."""
    # Create custom config if desired
    config = PairsTradingPipeline._get_default_config()
    config.update({
        'num_stocks': 30,  # Reduced for faster execution
        'train_test_split': 0.7,
        'entry_zscore': 2.0,
        'exit_zscore': 0.0,
        'transaction_cost': 0.001,
        'initial_capital': 100000,
        'output_dir': 'results'
    })
    
    pipeline = PairsTradingPipeline(config)
    pipeline.run_full_pipeline()
    
    # Print summary
    if pipeline.portfolio_results:
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("="*60)
        metrics = pipeline.portfolio_results['metrics']
        print(f"Total Return:         {metrics['Total_Return']:>10.2%}")
        print(f"Annualized Return:    {metrics['Annualized_Return']:>10.2%}")
        print(f"Volatility:           {metrics['Volatility']:>10.2%}")
        print(f"Sharpe Ratio:         {metrics['Sharpe_Ratio']:>10.2f}")
        print(f"Max Drawdown:         {metrics['Max_Drawdown']:>10.2%}")
        print("="*60 + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ['--report']:
        # Reuse saved prices and the screen to produce all assignment outputs.
        from build_research_report import main as report_main
        report_main(sys.argv[2:])
    else:
        main()
