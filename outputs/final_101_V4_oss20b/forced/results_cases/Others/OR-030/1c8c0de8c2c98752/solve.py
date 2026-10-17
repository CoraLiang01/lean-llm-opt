CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ '
          'column. The dealership aims to maximize total revenue using the initial inventory of car models classified '
          'under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are '
          'specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'BigMartSales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'FDK57',
                                         'operator': 'prefix',
                                         'value': 'FDK57'}],
                         'logic': 'and'},
             'original_rows': 5681,
             'records': [{'source_row': 1163,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1501,
                          'values': {'Demand': '40',
                                     'Initial Inventory': '100',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1576,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 1793,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.144'}},
                         {'source_row': 2438,
                          'values': {'Demand': '10',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.544'}},
                         {'source_row': 4297,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 4806,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '250',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.744'}},
                         {'source_row': 4942,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.844'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table['records']
    keys = [rec['source_row'] for rec in records]
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        key = rec['source_row']
        vals = rec['values']
        try:
            revenue[key] = float(vals['Revenue'])
            demand[key] = int(vals['Demand'])
            inventory[key] = int(vals['Initial Inventory'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid data for record {key}: {e}')
    for key in keys:
        if key not in revenue:
            raise ValueError(f'Revenue missing for key {key}')
        if key not in demand:
            raise ValueError(f'Demand missing for key {key}')
        if key not in inventory:
            raise ValueError(f'Initial Inventory missing for key {key}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in keys)), GRB.MAXIMIZE)
    for i in keys:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()