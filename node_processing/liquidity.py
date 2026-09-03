import numpy as np

def current_ratio(current_assets, current_liabilities):
    '''
    Can the firm meet short-term obligations
        - Red flag below 1.0

     - current_assets: act
     - current_liabilities: lct
    '''

    return current_assets/current_liabilities

def quick_ration(cash_equivalents, receivables, current_liabilities):
    '''
    Can the firm meet short-term obligations, excluding inventory
        - Red flag below 1.0

     - cash_equivalents: che
     - receivables: rect
     - current_liabilities: lct
    '''

    return (cash_equivalents + receivables)/current_liabilities

def cash_to_assets(cash_equivalents, total_assets):
    '''
    Cash on hand relative to the firms assets (proxy for firm size)

     - cash_equivalents: che
     - total_assets: at
    '''

    return cash_equivalents/total_assets

def wc_to_assets(working_capital, total_assets, current_assets, current_liabilities):
    '''
    Structural liquidity of the company
        - negative if illiquid

     - working_capital: wcap
     - total_assets: at
     - current_assets: act
     - current_liabilities: lct
    '''

    if working_capital == np.nan:
        return (current_assets - current_liabilities)/total_assets
    else:
        return working_capital/total_assets
    
