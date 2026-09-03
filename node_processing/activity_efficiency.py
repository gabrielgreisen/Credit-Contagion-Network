def asset_turnover(net_sales, total_assets):
    '''
    How efficiently the firm uses assets to generate revenue
     
     - net_sales: sale
     - total_assets: at
    '''

    return net_sales/total_assets

def receivables_turnover(net_sales, receivables):
    '''
    How quickly the firm collects payments (low values can signal customer distress)
     
     - net_sales: sale
     - total_assets: at
    '''

    return net_sales/receivables

def inventory_turnover(cogs, invt):
    '''
    How quickly inventory moves
        - Crucial for manufacturing and retail firms
     
     - cogs: cogs
     - invt: invt
    '''

    return cogs/invt