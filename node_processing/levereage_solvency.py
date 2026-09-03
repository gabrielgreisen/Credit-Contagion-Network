import numpy as np

def total_debt(long_term_debt, debt_current_liabilities):
    '''
     - long_term_debt: dltt
     - debt_current_liabilities: dlc
    '''

    total_debt = long_term_debt + debt_current_liabilities

    if total_debt == 0:
        return np.nan
    
    return total_debt

def debt_to_assets(total_debt, total_assets):
    '''
    How much of the firm is financed by debt

     - total_debt: total_debt
     - total_assets: at
    '''

    return total_debt / total_assets

def debt_to_equity(total_debt, common_equity):
    '''
    Leverage from equity holders' perspective

     - total_debt: total_debt
     - common_equity: ceq
    '''

    return total_debt/common_equity

def lt_debt_ratio(long_term_debt, total_assets):
    '''
    Long-term borrowing burden

     - long_term_debt: dltt
     - total_assets: at
    '''

    return long_term_debt, total_assets

def st_debt_ration(short_term_debt, total_assets):
    '''
    Short-term borrowing burden - reveals refinancing risk

     - short_term_debt: dlc
     - total_assets: at
    '''

    return short_term_debt/total_assets

def interest_coverage(ebitda, interest_expense):
    '''
    Shows if the firm can cover its interest from operations
        - Risky if below 1.0

     - ebitda: ebitda
     - interest_expense: xint
    '''

    return ebitda/interest_expense

def st_debt_share(short_term_debt, total_debt):
    '''
    Share of the debt maturing soon, which can cause rollover risk if too high

     - short_term_debt: dlc
     - total_debt: total_debt
    '''

    return short_term_debt/total_debt

