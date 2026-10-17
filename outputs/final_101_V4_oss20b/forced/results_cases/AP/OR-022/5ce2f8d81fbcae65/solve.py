CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'AP',
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
    cols = target_table.get('columns', [])
    required_cols = {'Product Name', 'Revenue', 'Demand', 'Initial Inventory'}
    if not required_cols.issubset(set(cols)):
        missing = required_cols - set(cols)
        raise ValueError(f"Missing required columns in table '{table_id}': {missing}")
    product_names = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in target_table.get('records', []):
        vals = rec.get('values', {})
        for col in required_cols:
            if col not in vals:
                raise ValueError(f"Record {rec.get('source_row')} missing column '{col}'")
        pname = vals['Product Name']
        if pname in product_names:
            raise ValueError(f"Duplicate product name detected: '{pname}'")
        product_names.append(pname)
        try:
            rev = float(vals['Revenue'])
            dem = int(vals['Demand'])
            inv = int(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Numeric conversion error in record {rec.get('source_row')}: {e}')
        revenue[pname] = rev
        demand[pname] = dem
        inventory[pname] = inv
    m = gurobipy.Model('max_revenue_27in')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, vtype=GRB.INTEGER, name='')
    for pname in product_names:
        m.addConstr(x[pname] <= demand[pname], name=f'demand_{pname}')
        m.addConstr(x[pname] <= inventory[pname], name=f'inventory_{pname}')
    m.setObjective(gurobipy.quicksum((revenue[pname] * x[pname] for pname in product_names)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()