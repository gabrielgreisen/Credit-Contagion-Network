import numpy as np


def _safe_div(numerator, denominator):
    if denominator is None or denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def market_value_equity(shares_outstanding, fiscal_year_end_stock_price):
    '''
    Market capitalization at fiscal year end

     - shares_outstanding: csho
     - fiscal_year_end_stock_price: prcc_f
    '''

    return shares_outstanding * fiscal_year_end_stock_price


def working_capital(current_assets, current_liabilities, wcap=None):
    '''
    Use reported wcap when available, else compute from act - lct

     - current_assets: act
     - current_liabilities: lct
     - wcap: wcap
    '''

    if wcap is not None and not np.isnan(wcap):
        return wcap
    return current_assets - current_liabilities


def altman_z(
    current_assets,
    current_liabilities,
    retained_earnings,
    ebit,
    shares_outstanding,
    fiscal_year_end_stock_price,
    total_liabilities,
    net_sales,
    total_assets,
    wcap=None,
):
    '''
    Altman (1968) Z-Score for publicly traded nonfinancial manufacturing firms.

        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        X1 = working capital / total assets
        X2 = retained earnings / total assets
        X3 = EBIT / total assets
        X4 = market value of equity / book value of total liabilities
        X5 = net sales / total assets

    Zones:
        Z > 2.99    -> safe
        1.81 < Z    -> grey
        Z < 1.81    -> distress

     - current_assets: act
     - current_liabilities: lct
     - retained_earnings: re
     - ebit: ebit (or oiadp as proxy)
     - shares_outstanding: csho
     - fiscal_year_end_stock_price: prcc_f
     - total_liabilities: lt
     - net_sales: sale
     - total_assets: at
     - wcap: wcap (preferred when available)
    '''

    wc = working_capital(current_assets, current_liabilities, wcap)
    mve = market_value_equity(shares_outstanding, fiscal_year_end_stock_price)

    x1 = _safe_div(wc, total_assets)
    x2 = _safe_div(retained_earnings, total_assets)
    x3 = _safe_div(ebit, total_assets)
    x4 = _safe_div(mve, total_liabilities)
    x5 = _safe_div(net_sales, total_assets)

    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def altman_z_prime(
    current_assets,
    current_liabilities,
    retained_earnings,
    ebit,
    common_equity,
    total_liabilities,
    net_sales,
    total_assets,
    wcap=None,
):
    '''
    Altman (1983) Z'-Score for private nonfinancial manufacturing firms.
    Substitutes book equity for market equity in X4.

        Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5

    Zones:
        Z' > 2.9    -> safe
        1.23 < Z'   -> grey
        Z' < 1.23   -> distress

     - current_assets: act
     - current_liabilities: lct
     - retained_earnings: re
     - ebit: ebit (or oiadp as proxy)
     - common_equity: ceq (book value of equity)
     - total_liabilities: lt
     - net_sales: sale
     - total_assets: at
     - wcap: wcap (preferred when available)
    '''

    wc = working_capital(current_assets, current_liabilities, wcap)

    x1 = _safe_div(wc, total_assets)
    x2 = _safe_div(retained_earnings, total_assets)
    x3 = _safe_div(ebit, total_assets)
    x4 = _safe_div(common_equity, total_liabilities)
    x5 = _safe_div(net_sales, total_assets)

    return 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5


def altman_z_double_prime(
    current_assets,
    current_liabilities,
    retained_earnings,
    ebit,
    common_equity,
    total_liabilities,
    total_assets,
    wcap=None,
):
    '''
    Altman (1995) Z''-Score for nonmanufacturers and emerging-market firms.
    Drops the asset-turnover (X5) term so the score is not biased by industry
    capital intensity. Use this for nonfinancial nonmanufacturers (services,
    tech, retail) — i.e. most of the universe outside SIC 2000-3999.

        Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    Zones:
        Z'' > 2.6    -> safe
        1.1 < Z''    -> grey
        Z'' < 1.1    -> distress

     - current_assets: act
     - current_liabilities: lct
     - retained_earnings: re
     - ebit: ebit (or oiadp as proxy)
     - common_equity: ceq (book value of equity)
     - total_liabilities: lt
     - total_assets: at
     - wcap: wcap (preferred when available)
    '''

    wc = working_capital(current_assets, current_liabilities, wcap)

    x1 = _safe_div(wc, total_assets)
    x2 = _safe_div(retained_earnings, total_assets)
    x3 = _safe_div(ebit, total_assets)
    x4 = _safe_div(common_equity, total_liabilities)

    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def z_zone(z, variant='z'):
    '''
    Map a Z-score to its distress zone using Altman's published cutoffs.

     - z: numeric score from altman_z / altman_z_prime / altman_z_double_prime
     - variant: 'z' (1968), 'z_prime' (1983), or 'z_double_prime' (1995)

    Returns 'safe', 'grey', 'distress', or np.nan if z is nan.
    '''

    if z is None or (isinstance(z, float) and np.isnan(z)):
        return np.nan

    cutoffs = {
        'z': (1.81, 2.99),
        'z_prime': (1.23, 2.90),
        'z_double_prime': (1.10, 2.60),
    }
    low, high = cutoffs[variant]

    if z < low:
        return 'distress'
    if z < high:
        return 'grey'
    return 'safe'


def is_nonfinancial(sic):
    '''
    Altman Z-Scores are not meaningful for financial firms (SIC 6000-6999).
    Use this guard before computing Z on a row.

     - sic: sic
    '''

    if sic is None or (isinstance(sic, float) and np.isnan(sic)):
        return False
    sic = int(sic)
    return not (6000 <= sic <= 6999)


def is_manufacturer(sic):
    '''
    Altman's original Z (1968) was fit on manufacturers (SIC 2000-3999).
    For nonmanufacturer nonfinancials, prefer altman_z_double_prime.

     - sic: sic
    '''

    if sic is None or (isinstance(sic, float) and np.isnan(sic)):
        return False
    sic = int(sic)
    return 2000 <= sic <= 3999
