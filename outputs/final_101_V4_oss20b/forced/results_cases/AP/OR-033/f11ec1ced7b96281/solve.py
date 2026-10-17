CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'AP',
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
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    I = []
    r_i = {}
    d_i = {}
    v_i = {}
    for rec in records:
        values = rec.get('values', {})
        product_name = values.get('Product Name')
        if product_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec}")
        if 'Baby' not in product_name:
            continue
        if 'Revenue' not in values or 'Demand' not in values or 'Initial Inventory' not in values:
            raise ValueError(f'Missing required columns in record {rec}')
        try:
            revenue = float(values['Revenue'])
            demand = int(values['Demand'])
            inventory = int(values['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid numeric conversion in record {rec}: {e}')
        I.append(product_name)
        r_i[product_name] = revenue
        d_i[product_name] = demand
        v_i[product_name] = inventory
    for i in I:
        if i not in r_i or i not in d_i or i not in v_i:
            raise ValueError(f'Missing parameter for product {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    ub_dict = {i: min(d_i[i], v_i[i]) for i in I}
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, ub=ub_dict, name='')
    model.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= d_i[i] for i in I), name='')
    model.addConstrs((x[i] <= v_i[i] for i in I), name='')
    model.optimize()
    return model
m = solve_problem()