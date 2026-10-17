CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'prefix',
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
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError("No records found in table 'file_0_view_0'")
    r = {}
    d = {}
    s = {}
    required_columns = {'Product Name', 'Revenue', 'Demand', 'Initial Inventory'}
    for rec in records:
        source_row = rec.get('source_row')
        values = rec.get('values', {})
        if not required_columns.issubset(values):
            missing = required_columns - values.keys()
            raise ValueError(f'Missing columns {missing} in record {source_row}')
        try:
            revenue = float(values['Revenue'])
            demand = float(values['Demand'])
            inventory = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Non-numeric value in record {source_row}: {e}')
        r[source_row] = revenue
        d[source_row] = demand
        s[source_row] = inventory
    indices = list(r.keys())
    ub_dict = {i: min(d[i], s[i]) for i in indices}
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(indices, lb=0, ub=ub_dict, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    for i in indices:
        m.addConstr(x[i] <= d[i], name=f'demand_{i}')
        m.addConstr(x[i] <= s[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()