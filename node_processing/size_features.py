import numpy as np

def log_assets(total_assets):
    '''
    Firm scale adjusteed for skewedness

     - total_assets: at
    '''

    return np.log(total_assets)

def log_revenue(net_sales):
    '''
    Operational scale

     - net_sales: sale
    '''

    return np.log(net_sales)

def log_mktcap(shares_outstanding, fiscal_year_end_stock_price):
    '''
    Market assesment of firm's value

     - shares_outstanding: csho
     - fiscal_year_end_stock_price: prcc_f
    '''

    return np.log(shares_outstanding * fiscal_year_end_stock_price)

def emp(n_employees):
    '''

     - n_employees: emp
    '''

    # Asses if an adjustment to firm size is worth it (represent fixed cost)


