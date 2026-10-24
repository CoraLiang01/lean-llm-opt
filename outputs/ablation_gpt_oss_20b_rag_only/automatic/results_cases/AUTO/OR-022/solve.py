CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
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
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    rev = {}
    dem = {}
    inv = {}
    I = []
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        if prod_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec.get('source_row')}")
        if prod_name in rev:
            raise ValueError(f"Duplicate product name '{prod_name}' found")
        try:
            rev_val = float(values.get('Revenue', ''))
            dem_val = float(values.get('Demand', ''))
            inv_val = float(values.get('Initial Inventory', ''))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('source_row')}: {e}")
        rev[prod_name] = rev_val
        dem[prod_name] = dem_val
        inv[prod_name] = inv_val
        I.append(prod_name)
    for i in I:
        if i not in rev or i not in dem or i not in inv:
            raise ValueError(f"Missing parameter for product '{i}'")
    m = gp.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= inv[i])
        m.addConstr(x[i] <= dem[i])
    m.optimize()
    return m
m = solve_problem()