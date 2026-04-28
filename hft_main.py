"""Main orchestrator for HFT RL Trading System"""

import argparse
from pathlib import Path
from rl_agent_trainer import HFTRLAgent, train_hft_agent
from hft_backtest import HFTBacktester
from live_hft_trader import LiveHFTTrader


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HFT RL Trading Agent System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'AAPL'],
                              help='Symbols to train on')
    train_parser.add_argument('--timesteps', type=int, default=500000,
                              help='Total training timesteps')
    train_parser.add_argument('--episodes', type=int, default=500,
                              help='Number of episodes')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest trained agent')
    backtest_parser.add_argument('--symbol', default='SPY', help='Symbol to backtest')
    backtest_parser.add_argument('--days', type=int, default=60, help='Days of data')
    backtest_parser.add_argument('--multi', action='store_true', help='Multi-symbol backtest')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--symbols', nargs='+', default=['SPY'],
                             help='Symbols to trade')
    live_parser.add_argument('--dry-run', action='store_true', help='Dry run (paper trading)')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Paper trading (simulation)')
    paper_parser.add_argument('--symbol', default='SPY', help='Symbol')
    paper_parser.add_argument('--days', type=int, default=30, help='Days to simulate')
    
    # Analyze patterns command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze patterns')
    analyze_parser.add_argument('--symbol', default='SPY', help='Symbol to analyze')
    analyze_parser.add_argument('--days', type=int, default=30, help='Days of data')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("\n" + "="*60)
        print("TRAINING HFT RL AGENT")
        print("="*60 + "\n")
        
        agent = HFTRLAgent()
        data = agent.prepare_training_data(args.symbols)
        
        if data:
            primary_symbol = list(data.keys())[0]
            df = data[primary_symbol]
            
            agent.create_environments(df)
            agent.build_model('PPO')
            agent.train(
                total_timesteps=args.timesteps,
                eval_frequency=10000
            )
            
            results = agent.evaluate(num_episodes=5)
            print("\nFinal Results:")
            print(f"  Avg Return: {results['avg_return']:.2%}")
            print(f"  Avg Sharpe: {results['avg_sharpe']:.2f}")
            print(f"  Win Rate: {results['avg_win_rate']:.1%}")
    
    elif args.command == 'backtest':
        print("\n" + "="*60)
        print("BACKTESTING HFT RL AGENT")
        print("="*60 + "\n")
        
        backtester = HFTBacktester()
        
        if args.multi:
            symbols = args.symbol.split(',') if ',' in args.symbol else ['SPY', 'QQQ', 'AAPL']
            results = backtester.run_multi_symbol_backtest(symbols, days=args.days)
        else:
            results = backtester.run_backtest(args.symbol, days=args.days)
            backtester.print_results(results)
            backtester.plot_results(results, f'backtest_{args.symbol}.png')
    
    elif args.command == 'live':
        print("\n" + "="*60)
        print("LIVE HFT TRADING")
        print("="*60 + "\n")
        
        trader = LiveHFTTrader(symbols=args.symbols)
        
        if args.dry_run:
            print("Running in DRY RUN mode (paper trading)\n")
        
        # Get signals for each symbol
        for symbol in args.symbols:
            df = trader.update_market_data(symbol)
            if not df.empty:
                signal = trader.generate_signal(symbol, df)
                print(f"\n{symbol} Signal:")
                action_labels = ['HOLD', 'BUY', 'SELL', 'CLOSE_ALL']
                print(f"  Action: {action_labels[signal.get('action', 0)]}")
                print(f"  Confidence: {signal.get('confidence', 0):.2f}")
                print(f"  Reason: {signal.get('reason', 'N/A')}")
        
        trader.print_summary()
    
    elif args.command == 'paper':
        print("\n" + "="*60)
        print("PAPER TRADING SIMULATION")
        print("="*60 + "\n")
        
        from data_provider import HFTDataProvider
        from hft_trading_env import HFTTradingEnv
        
        provider = HFTDataProvider()
        df = provider.get_historical_data(args.symbol, days=args.days)
        
        if not df.empty:
            env = HFTTradingEnv(df)
            obs = env.reset()
            
            print(f"Simulating {args.symbol} for {args.days} days...")
            
            for _ in range(min(1000, env.max_steps)):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                if done:
                    break
            
            stats = env.get_stats()
            print("\nSimulation Results:")
            print(f"  Final Value: ${env.capital:,.2f}")
            print(f"  Return: {stats['total_return']:.2%}")
            print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
            print(f"  Win Rate: {stats['win_rate']:.1%}")
            print(f"  Trades: {stats['total_trades']}")
    
    elif args.command == 'analyze':
        print("\n" + "="*60)
        print("PATTERN ANALYSIS")
        print("="*60 + "\n")
        
        from data_provider import HFTDataProvider
        from pattern_recognizer import PatternRecognizer
        from market_regime import MarketRegimeDetector
        
        provider = HFTDataProvider()
        df = provider.get_historical_data(args.symbol, days=args.days)
        
        if not df.empty:
            # Detect patterns
            recognizer = PatternRecognizer()
            patterns = recognizer.detect_patterns(df)
            
            print(f"\nDetected {len(patterns)} patterns in {args.symbol}:")
            
            pattern_counts = {}
            for p in patterns:
                pattern_counts[p.name] = pattern_counts.get(p.name, 0) + 1
            
            for name, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {count}")
            
            # Detect regime
            detector = MarketRegimeDetector()
            df_with_indicators = detector.add_indicators(df.copy())
            regime = detector.detect_regime(df_with_indicators)
            strength = detector.detect_regime_strength(df_with_indicators)
            
            print(f"\nCurrent Regime: {regime} (strength: {strength:.2f})")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
