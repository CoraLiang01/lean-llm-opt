CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'DifferentStoreSales.csv',
             'filters': {'conditions': [{'column': 'Product_Name',
                                         'dtype': 'string',
                                         'evidence': 'Books',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Books'}],
                         'logic': 'and'},
             'original_rows': 40,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1980',
                                     'Initial Inventory': '9920.0',
                                     'Product_Name': 'Books_15.15',
                                     'Revenue': '15.15'}},
                         {'source_row': 1,
                          'values': {'Demand': '3024',
                                     'Initial Inventory': '20160.0',
                                     'Product_Name': 'Books_30.3',
                                     'Revenue': '30.3'}},
                         {'source_row': 2,
                          'values': {'Demand': '4536',
                                     'Initial Inventory': '30000.0',
                                     'Product_Name': 'Books_45.45',
                                     'Revenue': '45.45'}},
                         {'source_row': 3,
                          'values': {'Demand': '5601',
                                     'Initial Inventory': '38360.0',
                                     'Product_Name': 'Books_60.6',
                                     'Revenue': '60.6'}},
                         {'source_row': 4,
                          'values': {'Demand': '7567',
                                     'Initial Inventory': '51450.0',
                                     'Product_Name': 'Books_75.75',
                                     'Revenue': '75.75'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    products = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        vals = rec.get('values', {})
        prod = vals.get('Product_Name')
        if prod is None:
            raise ValueError(f'Missing Product_Name in record {rec.get('source_row')}')
        rev_str = vals.get('Revenue')
        dem_str = vals.get('Demand')
        inv_str = vals.get('Initial Inventory')
        if rev_str is None or dem_str is None or inv_str is None:
            raise ValueError(f'Missing parameter in record {rec.get('source_row')} for product {prod}')
        try:
            rev = float(rev_str)
            dem = float(dem_str)
            inv = float(inv_str)
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {rec.get('source_row')} for product {prod}: {e}')
        products.append(prod)
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
    for prod in products:
        if prod not in revenue or prod not in demand or prod not in inventory:
            raise ValueError(f'Missing parameter for product {prod}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
    for i in products:
        model.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
        model.addConstr(x[i] <= demand[i], name=f'demand_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()