CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
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
    rev = {}
    dem = {}
    inv = {}
    I = set()
    for rec in records:
        vals = rec.get('values', {})
        prod = vals.get('Product Name', '').strip()
        if not prod:
            continue
        I.add(prod)
        try:
            rev_val = float(vals.get('Revenue', 0))
            dem_val = float(vals.get('Demand', 0))
            inv_val = float(vals.get('Initial Inventory', 0))
        except ValueError as e:
            raise ValueError(f'Non-numeric value in record {rec}') from e
        if rev_val < 0 or dem_val < 0 or inv_val < 0:
            raise ValueError(f'Negative parameter in record {rec}')
        rev[prod] = rev_val
        dem[prod] = dem_val
        inv[prod] = inv_val
    if not I:
        raise ValueError('No products found after filtering')
    for i in I:
        if i not in rev or i not in dem or i not in inv:
            raise ValueError(f'Missing parameter for product {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= dem[i])
        model.addConstr(x[i] <= inv[i])
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(model.ObjVal)
        for var in model.getVars():
            print(var.VarName, var.X)
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()