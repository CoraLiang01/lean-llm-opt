CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': 'ZZ',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'ZZ'}],
                         'logic': 'and'},
             'original_rows': 5242,
             'records': [{'source_row': 5237,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}},
                         {'source_row': 5238,
                          'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}},
                         {'source_row': 5239,
                          'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}},
                         {'source_row': 5240,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}},
                         {'source_row': 5241,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Table file_0_view_0 not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError('No records in table file_0_view_0')
    rev = {}
    dem = {}
    inv = {}
    skus = []
    for rec in records:
        values = rec.get('values', {})
        sku = values.get('SKU')
        if sku is None:
            raise ValueError('Missing SKU in record')
        rev_val = values.get('Revenue')
        dem_val = values.get('Demand')
        inv_val = values.get('Initial Inventory')
        if rev_val is None or dem_val is None or inv_val is None:
            raise ValueError(f'Missing parameter in record for SKU {sku}')
        try:
            rev_val = float(rev_val)
            dem_val = float(dem_val)
            inv_val = float(inv_val)
        except ValueError:
            raise ValueError(f'Non-numeric parameter in record for SKU {sku}')
        rev[sku] = rev_val
        dem[sku] = dem_val
        inv[sku] = inv_val
        skus.append(sku)
    m = gurobipy.Model('MaxRevenueZZ')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(skus, vtype=GRB.INTEGER, name='')
    for sku in skus:
        m.addConstr(x[sku] <= dem[sku], name=f'dem_{sku}')
        m.addConstr(x[sku] <= inv[sku], name=f'inv_{sku}')
    m.setObjective(gurobipy.quicksum((rev[sku] * x[sku] for sku in skus)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()