def roa(net_income, total_assets):
    '''
    Profitability normalized by size

     - net_income: ni
     - total_assets: at
    '''

    return net_income/total_assets

def roe(net_income, common_equity):
    '''
    Profitability from shareholder perspective

     - net_income: ni
     - common_equity: ceq
    '''

    return net_income/common_equity

def ebitda_margin(ebitda, net_sales):
    '''
    Operating profitability

     - ebitda: ebitda
     - net_sales: sale
    '''

    return ebitda/net_sales

def gross_margin(gross_profit, net_sales):
    '''
    Revenue minus COGS, measuring pricing power

     - gross_prfit: gp
     - net_sales: sale
    '''

    return gross_profit/net_sales

def operational_margin(operating_income_after_depreciation, net_sales):
    '''
    Profitability from core operations

     - operating income_after_depreciation: oiadp
     - net_sales: sale
    '''

    return operating_income_after_depreciation/net_sales

def ni_change(ni_t, ni_t_1):

    return ni_t - ni_t_1