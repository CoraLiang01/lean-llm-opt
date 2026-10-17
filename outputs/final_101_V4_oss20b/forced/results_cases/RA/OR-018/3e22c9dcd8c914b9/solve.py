CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
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
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError("No records found in table 'file_0_view_0'")
    I = []
    revenue = {}
    demand = {}
    initial_inventory = {}
    for rec in records:
        idx = rec.get('source_row')
        if idx is None:
            raise ValueError("Record missing 'source_row' key")
        I.append(idx)
        vals = rec.get('values', {})
        for key in ('Revenue', 'Demand', 'Initial Inventory'):
            if key not in vals:
                raise ValueError(f"Missing '{key}' in record with source_row {idx}")
        try:
            rev = float(vals['Revenue'])
        except Exception as e:
            raise ValueError(f'Invalid Revenue value for source_row {idx}: {e}')
        try:
            dem = int(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Invalid Demand value for source_row {idx}: {e}')
        try:
            inv = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid Initial Inventory value for source_row {idx}: {e}')
        revenue[idx] = rev
        demand[idx] = dem
        initial_inventory[idx] = inv
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= demand[i])
        model.addConstr(x[i] <= initial_inventory[i])
    model.optimize()
    return model
m = solve_problem()