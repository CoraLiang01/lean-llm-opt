CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘ZZ’',
                                         'operator': 'prefix',
                                         'value': 'ZZ'}],
                         'logic': 'and'},
             'original_rows': 5242,
             'records': [{'source_row': 5237,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}},
                         {'source_row': 5238,
                          'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}},
                         {'source_row': 5239,
                          'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}},
                         {'source_row': 5240,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}},
                         {'source_row': 5241,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}],
             'returned_rows': 5,
             'role': 'product revenue, demand, and inventory data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    zz_rows = []
    for (source_row, row) in frame.iterrows():
        sku = row['SKU']
        if isinstance(sku, str) and sku.casefold().startswith('zz'):
            zz_rows.append((source_row, row))
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in zz_rows:
        sku = row['SKU']
        try:
            A_i = float(row['Revenue'])
            d_i = float(row['Demand'])
            s_i = float(row['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Non-numeric coefficient for SKU {sku}: {e}')
        I.append(sku)
        revenue[sku] = A_i
        demand[sku] = d_i
        inventory[sku] = s_i
    if not set(I) == set(revenue) == set(demand) == set(inventory):
        raise ValueError('Mismatch in index sets for coefficients.')
    ub = {sku: min(demand[sku], inventory[sku]) for sku in I}
    m = gp.Model('RetailStore_ZZ_Revenue_Max')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(I, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[sku] * quantity_vars[sku] for sku in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)