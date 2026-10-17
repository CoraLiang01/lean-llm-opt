CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for '
          'these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant '
          'aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, '
          'which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, '
          'and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to '
          'be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The '
          'variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to '
          'fulfill.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Aalop',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Aalop'}],
                         'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    ids = [rec['source_row'] for rec in records]
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        i = rec['source_row']
        revenue[i] = float(rec['values']['Revenue'])
        demand[i] = int(rec['values']['Demand'])
        inventory[i] = float(rec['values']['Initial Inventory'])
    x = m.addVars(ids, vtype=GRB.INTEGER, name='')
    for i in ids:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
        m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()