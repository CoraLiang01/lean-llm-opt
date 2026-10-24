CSVQA_DATA = {'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘ZZ’ product i that the store plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RetailStoreSalesTransactions(ScannerData).csv', 'records': [{'data_row_number': 5237, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}}, {'data_row_number': 5238, 'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}}, {'data_row_number': 5239, 'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}}, {'data_row_number': 5240, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}}, {'data_row_number': 5241, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    I = []
    r_i = {}
    d_i = {}
    inv_i = {}
    for rec in records:
        values = rec['values']
        sku = values['SKU']
        if sku in I:
            raise ValueError(f'Duplicate SKU {sku}')
        I.append(sku)
        try:
            r_i[sku] = float(values['Revenue'])
            d_i[sku] = int(values['Demand'])
            inv_i[sku] = float(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing field {e} in record {rec}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    for i in I:
        m.addConstr(x[i] <= d_i[i], name='')
        m.addConstr(x[i] <= inv_i[i], name='')
    m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
