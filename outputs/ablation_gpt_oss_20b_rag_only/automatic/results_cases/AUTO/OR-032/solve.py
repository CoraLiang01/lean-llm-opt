CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
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
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = None
    for t in tables:
        if t.get('table_id') == table_id:
            table = t
            break
    if table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    I = []
    revenue = {}
    demand = {}
    inv = {}
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product_Name')
        if prod_name is None:
            raise ValueError(f"Missing Product_Name in record {rec.get('source_row')}")
        if prod_name in I:
            raise ValueError(f"Duplicate Product_Name '{prod_name}' in records")
        I.append(prod_name)
        rev_val = values.get('Revenue')
        dem_val = values.get('Demand')
        inv_val = values.get('Initial Inventory')
        if rev_val is None or dem_val is None or inv_val is None:
            raise ValueError(f"Missing parameter in record for product '{prod_name}'")
        try:
            rev = float(rev_val)
        except ValueError:
            raise ValueError(f"Invalid Revenue value '{rev_val}' for product '{prod_name}'")
        try:
            dem = int(dem_val)
        except ValueError:
            raise ValueError(f"Invalid Demand value '{dem_val}' for product '{prod_name}'")
        try:
            inv_val_f = float(inv_val)
        except ValueError:
            raise ValueError(f"Invalid Initial Inventory value '{inv_val}' for product '{prod_name}'")
        revenue[prod_name] = rev
        demand[prod_name] = dem
        inv[prod_name] = inv_val_f
    for prod in I:
        if prod not in revenue or prod not in demand or prod not in inv:
            raise ValueError(f"Missing parameters for product '{prod}'")
    model = gp.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gp.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= demand[i])
        model.addConstr(x[i] <= inv[i])
    model.optimize()
    return model
m = solve_problem()