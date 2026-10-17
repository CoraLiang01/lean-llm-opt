CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the '
          '‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective '
          'is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. '
          'The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. '
          'During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n'
          '\n'
          'Demand for each product during the sales period is assumed to be deterministic and known in advance, with '
          'demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of '
          'units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative '
          'integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the '
          'fulfillment quantities must satisfy both inventory and demand constraints.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': 'id999',
                                         'operator': 'prefix',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
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
    revenue = {}
    demand = {}
    inventory = {}
    product_ids = []
    for rec in records:
        values = rec.get('values', {})
        for col in ('id_number', 'Revenue', 'Demand', 'Initial Inventory'):
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {rec.get('source_row')}")
        id_num = values['id_number']
        if not isinstance(id_num, str) or not id_num.startswith('id999'):
            continue
        try:
            rev = float(values['Revenue'])
            dem = int(values['Demand'])
            inv = int(values['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid numeric conversion for record {rec.get('source_row')}: {e}')
        product_ids.append(id_num)
        revenue[id_num] = rev
        demand[id_num] = dem
        inventory[id_num] = inv
    if not product_ids:
        raise ValueError("No products with id prefix 'id999' found after filtering")
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in product_ids)), GRB.MAXIMIZE)
    for i in product_ids:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()