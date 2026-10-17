CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'format': None,
                                         'inclusive': 'both',
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
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    view_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            view_table = tbl
            break
    if view_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = view_table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        if prod_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec.get('source_row')}")
        if prod_name in rev:
            raise ValueError(f"Duplicate product name '{prod_name}' in records")
        try:
            rev_val = float(values['Revenue'])
            dem_val = float(values['Demand'])
            inv_val = float(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing column {e} in record {rec.get('source_row')}')
        except ValueError as e:
            raise ValueError(f'Non-numeric value in record {rec.get('source_row')}: {e}')
        rev[prod_name] = rev_val
        dem[prod_name] = dem_val
        inv[prod_name] = inv_val
    m = gurobipy.Model('Baby_Product_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    prod_names = list(rev.keys())
    x = m.addVars(prod_names, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in prod_names)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= dem[i] for i in prod_names), name='')
    m.addConstrs((x[i] <= inv[i] for i in prod_names), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()