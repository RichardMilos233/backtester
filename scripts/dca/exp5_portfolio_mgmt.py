"""EXP5 driver: runs parts A, B, C (see findings.md in docs/dca/exp5_portfolio_mgmt/)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp5a_transition, exp5b_rebalance, exp5c_satellite

if __name__ == '__main__':
    exp5a_transition.main()
    exp5b_rebalance.main()
    exp5c_satellite.main()
