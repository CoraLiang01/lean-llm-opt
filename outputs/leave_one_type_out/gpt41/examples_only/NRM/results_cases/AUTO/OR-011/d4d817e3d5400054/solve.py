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
                                         'evidence': "'id999' products",
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
from gurobipy import Model, GRB

def solve_supermarket_fulfillment(CSVQA_DATA):
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == table_id:
            table = t
            break
    if table is None:
        raise RuntimeError('Required table_id not found in CSVQA_DATA.')
    I = []
    Revenue = {}
    InitialInventory = {}
    Demand = {}
    for rec in table['records']:
        vals = rec['values']
        if 'id_number' not in vals or vals['id_number'] != 'id999':
            continue
        idx = rec['values']['id_number'] + '_' + str(rec['source_row'])
        I.append(idx)
        try:
            Revenue[idx] = float(vals['Revenue'])
        except Exception:
            raise ValueError(f'Missing or invalid Revenue for {idx}')
        try:
            InitialInventory[idx] = int(vals['Initial Inventory'])
        except Exception:
            raise ValueError(f'Missing or invalid Initial Inventory for {idx}')
        try:
            Demand[idx] = int(vals['Demand'])
        except Exception:
            raise ValueError(f'Missing or invalid Demand for {idx}')
    if not I:
        raise RuntimeError("No products with id_number == 'id999' found.")
    for idx in I:
        if idx not in Revenue or idx not in InitialInventory or idx not in Demand:
            raise RuntimeError(f'Missing data for product {idx}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((Revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= InitialInventory[i] for i in I), name='')
    m.addConstrs((x[i] <= Demand[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName}: {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_supermarket_fulfillment(CSVQA_DATA)