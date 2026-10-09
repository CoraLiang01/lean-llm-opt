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
                                         'inclusive': 'both',
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
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = [{'SKU': 'ZZ2AO', 'Revenue': '24.38', 'Demand': '2', 'Initial Inventory': '10.0'}, {'SKU': 'ZZDW7', 'Revenue': '30.12', 'Demand': '4', 'Initial Inventory': '20.0'}, {'SKU': 'ZZM1A', 'Revenue': '19.52', 'Demand': '82', 'Initial Inventory': '530.0'}, {'SKU': 'ZZNC5', 'Revenue': '10.79', 'Demand': '2', 'Initial Inventory': '10.0'}, {'SKU': 'ZZX6K', 'Revenue': '111.81', 'Demand': '2', 'Initial Inventory': '10.0'}]
    I = [rec['SKU'] for rec in data]
    revenue = {}
    demand = {}
    inventory = {}
    for rec in data:
        sku = rec['SKU']
        try:
            revenue[sku] = float(rec['Revenue'])
            demand[sku] = int(rec['Demand'])
            inventory[sku] = float(rec['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for SKU {sku}: {e}')
    for i in I:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for SKU {i}')
    ub = {i: min(demand[i], inventory[i]) for i in I}
    m = gp.Model('RetailStore_ZZ_MaxRevenue')
    x = m.addVars(I, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()