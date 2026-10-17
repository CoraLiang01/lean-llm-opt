CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [{'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': 'Organ',
                                         'operator': 'contains',
                                         'value': 'Organ'}],
                         'logic': 'and'},
             'original_rows': 23,
             'records': [{'source_row': 17,
                          'values': {'Demand': '678906',
                                     'Initial Inventory': '5034020.0',
                                     'Revenue': '60.8',
                                     'Sub Category': 'Organic Fruits'}},
                         {'source_row': 18,
                          'values': {'Demand': '749927',
                                     'Initial Inventory': '5589290.0',
                                     'Revenue': '918.45',
                                     'Sub Category': 'Organic Staples'}},
                         {'source_row': 19,
                          'values': {'Demand': '699808',
                                     'Initial Inventory': '5202710.0',
                                     'Revenue': '77.52',
                                     'Sub Category': 'Organic Vegetables'}}],
             'returned_rows': 3,
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
    I = []
    rev_i = {}
    dem_i = {}
    inv_i = {}
    for rec in records:
        values = rec.get('values', {})
        key = values.get('Sub Category')
        if key is None:
            raise ValueError(f"Missing 'Sub Category' in record {rec.get('source_row')}.")
        if key in I:
            raise ValueError(f"Duplicate index key '{key}' found.")
        I.append(key)
        rev_val = values.get('Revenue')
        dem_val = values.get('Demand')
        inv_val = values.get('Initial Inventory')
        if rev_val is None or dem_val is None or inv_val is None:
            raise ValueError(f"Missing parameter in record for key '{key}'.")
        try:
            rev_i[key] = float(rev_val)
            dem_i[key] = int(float(dem_val))
            inv_i[key] = float(inv_val)
        except ValueError as e:
            raise ValueError(f"Invalid numeric value for key '{key}': {e}")
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((rev_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= inv_i[i], name=f'inv_{i}')
        m.addConstr(x[i] <= dem_i[i], name=f'dem_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()