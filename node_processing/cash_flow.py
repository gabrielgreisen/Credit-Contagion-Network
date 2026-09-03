def opcf_to_assets(opcf, at):
    '''
    Cash generation from operations relative to size
     
     - opcf: oancf
     - total_assets: at
    '''

    return opcf/at

def capex_to_assets(capx, at):
    '''
    Investment intensity
     
     - capx: capx
     - total_assets: at
    '''

    return capx/at

def fcf_to_assets(opcf, capx, at):
    '''
    Free cash flow available after maintaining operations relative to size
        - Negative: burning cash
     
     - opcf: oancf
     - capx: capx
     - total_assets: at
    '''

    return (opcf - capx)/at