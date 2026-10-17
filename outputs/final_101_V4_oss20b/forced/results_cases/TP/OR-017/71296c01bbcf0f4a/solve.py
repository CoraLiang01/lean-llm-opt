CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': 'ZZ',
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
    if not tables:
        raise ValueError('No tables found in CSVQA_DATA')
    table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records in the required table')
    I = []
    r_i = {}
    d_i = {}
    s_i = {}
    for rec in records:
        values = rec.get('values', {})
        for col in ('SKU', 'Revenue', 'Demand', 'Initial Inventory'):
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {rec.get('source_row')}")
        sku = values['SKU']
        if sku in I:
            raise ValueError(f"Duplicate SKU '{sku}' found in records")
        I.append(sku)
        try:
            r_i[sku] = float(values['Revenue'])
            d_i[sku] = float(values['Demand'])
            s_i[sku] = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {rec.get('source_row')}: {e}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    for i in I:
        model.addConstr(x[i] <= d_i[i])
        model.addConstr(x[i] <= s_i[i])
    model.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for variable in model.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()