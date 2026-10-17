CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'TP',
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
    tables = CSVQA_DATA.get('tables', [])
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
    revenue_dict = {}
    demand_dict = {}
    inventory_dict = {}
    index_set = []
    for rec in records:
        values = rec.get('values', {})
        product = values.get('Product Name')
        if product is None:
            raise ValueError(f"Missing 'Product Name' in record {rec.get('source_row')}")
        try:
            revenue = float(values.get('Revenue', '0'))
        except ValueError:
            raise ValueError(f"Invalid Revenue value for product '{product}'")
        try:
            demand = int(values.get('Demand', '0'))
        except ValueError:
            raise ValueError(f"Invalid Demand value for product '{product}'")
        try:
            inventory = int(values.get('Initial Inventory', '0'))
        except ValueError:
            raise ValueError(f"Invalid Initial Inventory value for product '{product}'")
        revenue_dict[product] = revenue
        demand_dict[product] = demand
        inventory_dict[product] = inventory
        index_set.append(product)
    for prod in index_set:
        if prod not in revenue_dict:
            raise ValueError(f"Revenue missing for product '{prod}'")
        if prod not in demand_dict:
            raise ValueError(f"Demand missing for product '{prod}'")
        if prod not in inventory_dict:
            raise ValueError(f"Inventory missing for product '{prod}'")
    m = gurobipy.Model('MaxRevenue27in')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(index_set, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in index_set)), GRB.MAXIMIZE)
    for i in index_set:
        m.addConstr(x[i] <= demand_dict[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory_dict[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()