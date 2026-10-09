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
 'route': 'NRM',
 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': "'id999'",
                                         'operator': 'exact',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
import pandas as pd
from gurobipy import Model, GRB, quicksum
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_index': 0, 'file_name': 'OnlineRetailSalesDataset.csv', 'filters': {'conditions': [{'column': 'id_number', 'dtype': 'string', 'evidence': "'id999'", 'operator': 'exact', 'value': 'id999'}], 'logic': 'and'}, 'original_rows': 900, 'records': [{'source_row': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'returned_rows': 1, 'role': 'product revenue, demand, and inventory parameters', 'table_id': 'file_0_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
I = []
r = {}
d = {}
s = {}
for rec in table['records']:
    vals = rec['values']
    prod_id = vals['id_number']
    I.append(prod_id)
    try:
        r[prod_id] = float(vals['Revenue'])
        d[prod_id] = int(vals['Demand'])
        s[prod_id] = int(vals['Initial Inventory'])
    except Exception as e:
        raise RuntimeError(f'Error parsing numeric fields for product {prod_id}: {e}')
for i in I:
    if i not in r or i not in d or i not in s:
        raise RuntimeError(f'Missing parameter for product {i}.')

def build_and_solve(I, r, d, s):
    m = Model()
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=None, name='')
    inv_constrs = m.addConstrs((x_vars[i] <= s[i] for i in I), name='')
    dem_constrs = m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve(I, r, d, s)