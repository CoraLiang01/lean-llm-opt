CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'AP',
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
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    I = []
    rev_dict = {}
    dem_dict = {}
    inv_dict = {}
    for rec in records:
        values = rec.get('values', {})
        sku = values.get('SKU')
        if sku is None:
            continue
        try:
            rev = float(values.get('Revenue', 0))
            dem = int(values.get('Demand', 0))
            inv = float(values.get('Initial Inventory', 0))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid data types for SKU '{sku}'")
        I.append(sku)
        rev_dict[sku] = rev
        dem_dict[sku] = dem
        inv_dict[sku] = inv
    for sku in I:
        if sku not in rev_dict:
            raise ValueError(f"Missing revenue for SKU '{sku}'")
        if sku not in dem_dict:
            raise ValueError(f"Missing demand for SKU '{sku}'")
        if sku not in inv_dict:
            raise ValueError(f"Missing inventory for SKU '{sku}'")
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((rev_dict[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= inv_dict[i], name='')
        model.addConstr(x[i] <= dem_dict[i], name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()