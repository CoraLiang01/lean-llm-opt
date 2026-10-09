CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': "'products classified under ‘ZZ’'",
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
             'role': 'product revenue, demand, and inventory parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum

def build_and_solve_model():
    import re
    CSVQA_DATA = {'ignored_file_indices': [], 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘ZZ’ product i that the store plans to fulfill.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'], 'file_index': 0, 'file_name': 'RetailStoreSalesTransactions(ScannerData).csv', 'filters': {'conditions': [{'column': 'SKU', 'dtype': 'string', 'evidence': "'products classified under ‘ZZ’'", 'inclusive': 'both', 'operator': 'prefix', 'value': 'ZZ'}], 'logic': 'and'}, 'original_rows': 5242, 'records': [{'source_row': 5237, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}}, {'source_row': 5238, 'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}}, {'source_row': 5239, 'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}}, {'source_row': 5240, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}}, {'source_row': 5241, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}], 'returned_rows': 5, 'role': 'product revenue, demand, and inventory parameters', 'table_id': 'file_0_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == table_id:
            table = t
            break
    if table is None:
        raise RuntimeError('Required table_id not found in CSVQA_DATA.')
    I = []
    r = {}
    s = {}
    d = {}
    for rec in table['records']:
        vals = rec['values']
        sku = vals['SKU']
        if not re.match('^ZZ', sku):
            continue
        I.append(sku)
        try:
            r[sku] = float(vals['Revenue'])
        except Exception:
            raise ValueError(f'Revenue missing or invalid for SKU {sku}')
        try:
            s[sku] = float(vals['Initial Inventory'])
        except Exception:
            raise ValueError(f'Initial Inventory missing or invalid for SKU {sku}')
        try:
            d[sku] = float(vals['Demand'])
        except Exception:
            raise ValueError(f'Demand missing or invalid for SKU {sku}')
    for sku in I:
        if sku not in r or sku not in s or sku not in d:
            raise RuntimeError(f'Missing parameter for SKU {sku}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub={sku: min(int(s[sku]), int(d[sku])) for sku in I}, name='')
    m.setObjective(quicksum((r[sku] * x_vars[sku] for sku in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for sku in I:
            print(x_vars[sku].VarName, x_vars[sku].X)
    else:
        print(m.Status)
    return m
m = build_and_solve_model()