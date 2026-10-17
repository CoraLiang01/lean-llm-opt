CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
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
    target_table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == target_table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{target_table_id}' not found in CSVQA_DATA.")
    records = target_table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{target_table_id}'.")
    revenue = {}
    demand = {}
    inventory = {}
    product_names = []
    for rec in records:
        values = rec.get('values', {})
        for col in ('Product_Name', 'Revenue', 'Demand', 'Initial Inventory'):
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {rec.get('source_row')}.")
        prod = values['Product_Name']
        if prod in revenue:
            raise ValueError(f"Duplicate product name '{prod}' found.")
        try:
            rev = float(values['Revenue'])
        except Exception as e:
            raise ValueError(f"Invalid revenue value for product '{prod}': {values['Revenue']}") from e
        try:
            dem = int(values['Demand'])
        except Exception as e:
            raise ValueError(f"Invalid demand value for product '{prod}': {values['Demand']}") from e
        try:
            inv = float(values['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Invalid inventory value for product '{prod}': {values['Initial Inventory']}") from e
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
        product_names.append(prod)
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= demand[i] for i in product_names), name='')
    model.addConstrs((x[i] <= inventory[i] for i in product_names), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()