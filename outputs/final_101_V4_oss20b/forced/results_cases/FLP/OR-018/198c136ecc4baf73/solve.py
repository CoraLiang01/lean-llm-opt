CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'FLP',
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
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError("No records returned for table 'file_0_view_0'")
    I = set()
    R_i = {}
    D_i = {}
    I_i = {}
    for rec in records:
        values = rec.get('values', {})
        try:
            prod_name = values['Product Name']
            revenue = float(values['Revenue'])
            demand = float(values['Demand'])
            init_inv = float(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing required field {e} in record {rec.get('source_row')}')
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {rec.get('source_row')}: {e}')
        I.add(prod_name)
        R_i[prod_name] = revenue
        D_i[prod_name] = demand
        I_i[prod_name] = init_inv
    for i in I:
        if i not in R_i:
            raise ValueError(f'Missing revenue for product {i}')
        if i not in D_i:
            raise ValueError(f'Missing demand for product {i}')
        if i not in I_i:
            raise ValueError(f'Missing initial inventory for product {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((R_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= D_i[i] for i in I), name='')
    model.addConstrs((x[i] <= I_i[i] for i in I), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()