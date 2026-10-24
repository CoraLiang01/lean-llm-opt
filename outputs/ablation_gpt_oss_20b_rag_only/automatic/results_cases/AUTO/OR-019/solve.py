CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    I = []
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        values = rec.get('values', {})
        product_name = values.get('Product Name')
        if product_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec.get('source_row')}")
        if product_name in I:
            raise ValueError(f"Duplicate product name '{product_name}' found")
        I.append(product_name)
        try:
            rev_val = float(values.get('Revenue', ''))
            dem_val = float(values.get('Demand', ''))
            inv_val = float(values.get('Initial Inventory', ''))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('source_row')}: {e}")
        rev[product_name] = rev_val
        dem[product_name] = dem_val
        inv[product_name] = inv_val
    for i in I:
        if i not in rev:
            raise ValueError(f"Missing revenue for product '{i}'")
        if i not in dem:
            raise ValueError(f"Missing demand for product '{i}'")
        if i not in inv:
            raise ValueError(f"Missing inventory for product '{i}'")
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    for i in I:
        model.addConstr(x[i] <= dem[i])
        model.addConstr(x[i] <= inv[i])
    model.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()