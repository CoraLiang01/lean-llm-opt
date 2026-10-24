CSVQA_DATA = {'query': 'A retail company wants to decide where to open warehouses to minimize total costs while meeting the demands of several stores in a region. Each potential warehouse has an opening cost, and there is a known transportation cost for supplying goods to each store from each warehouse. Each warehouse has a capacity, and each store has a specific demand. The potential warehouses costs are in PotentialWarehouses_Costs.csv, demand for each store is in Stores_Demands.csv and the transportation costs c_ij from warehouse i to store j is in TransportationCost.csv\n\n    You are to determine which warehouses to open and how to assign the demand of each store to the warehouses such that all store demands are met, warehouse capacities are not exceeded, and the total cost (opening + transportation) is minimized.', 'relationships': [], 'tables': [{'columns': ['Warehouse (i)', 'Opening Cost (fi)', 'Capacity (units)'], 'file_name': 'PotentialWarehouses_Costs.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '3000', 'Warehouse (i)': '1'}}, {'data_row_number': 1, 'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '3200', 'Warehouse (i)': '2'}}, {'data_row_number': 2, 'values': {'Capacity (units)': '200', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '3'}}, {'data_row_number': 3, 'values': {'Capacity (units)': '150', 'Opening Cost (fi)': '2800', 'Warehouse (i)': '4'}}, {'data_row_number': 4, 'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3500', 'Warehouse (i)': '5'}}, {'data_row_number': 5, 'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2700', 'Warehouse (i)': '6'}}, {'data_row_number': 6, 'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '2900', 'Warehouse (i)': '7'}}, {'data_row_number': 7, 'values': {'Capacity (units)': '175', 'Opening Cost (fi)': '3050', 'Warehouse (i)': '8'}}, {'data_row_number': 8, 'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '9'}}, {'data_row_number': 9, 'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '2200', 'Warehouse (i)': '10'}}, {'data_row_number': 10, 'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2890', 'Warehouse (i)': '11'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Store (j)', 'Demand (units, dj)'], 'file_name': 'Stores_Demands.csv', 'records': [{'data_row_number': 0, 'values': {'Demand (units, dj)': '30', 'Store (j)': '1'}}, {'data_row_number': 1, 'values': {'Demand (units, dj)': '40', 'Store (j)': '2'}}, {'data_row_number': 2, 'values': {'Demand (units, dj)': '20', 'Store (j)': '3'}}, {'data_row_number': 3, 'values': {'Demand (units, dj)': '35', 'Store (j)': '4'}}, {'data_row_number': 4, 'values': {'Demand (units, dj)': '20', 'Store (j)': '5'}}, {'data_row_number': 5, 'values': {'Demand (units, dj)': '25', 'Store (j)': '6'}}, {'data_row_number': 6, 'values': {'Demand (units, dj)': '45', 'Store (j)': '7'}}, {'data_row_number': 7, 'values': {'Demand (units, dj)': '38', 'Store (j)': '8'}}, {'data_row_number': 8, 'values': {'Demand (units, dj)': '32', 'Store (j)': '9'}}, {'data_row_number': 9, 'values': {'Demand (units, dj)': '41', 'Store (j)': '10'}}, {'data_row_number': 10, 'values': {'Demand (units, dj)': '44', 'Store (j)': '11'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11'], 'file_name': 'TransportationCost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'W1', 'W1': '12', 'W10': '14', 'W11': '15', 'W2': '11', 'W3': '14', 'W4': '15', 'W5': '17', 'W6': '13', 'W7': '12', 'W8': '16', 'W9': '16'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'W2', 'W1': '17', 'W10': '15', 'W11': '16', 'W2': '19', 'W3': '15', 'W4': '20', 'W5': '18', 'W6': '14', 'W7': '17', 'W8': '15', 'W9': '13'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'W3', 'W1': '13', 'W10': '18', 'W11': '17', 'W2': '14', 'W3': '12', 'W4': '14', 'W5': '16', 'W6': '15', 'W7': '11', 'W8': '14', 'W9': '16'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'W4', 'W1': '18', 'W10': '13', 'W11': '18', 'W2': '16', 'W3': '17', 'W4': '13', 'W5': '18', 'W6': '17', 'W7': '14', 'W8': '19', 'W9': '16'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'W5', 'W1': '10', 'W10': '15', 'W11': '17', 'W2': '13', 'W3': '12', 'W4': '19', 'W5': '15', 'W6': '11', 'W7': '12', 'W8': '14', 'W9': '12'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'W6', 'W1': '15', 'W10': '18', 'W11': '19', 'W2': '12', 'W3': '14', 'W4': '16', 'W5': '13', 'W6': '17', 'W7': '16', 'W8': '16', 'W9': '14'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'W7', 'W1': '14', 'W10': '16', 'W11': '14', 'W2': '13', 'W3': '15', 'W4': '17', 'W5': '12', 'W6': '13', 'W7': '14', 'W8': '15', 'W9': '12'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'W8', 'W1': '19', 'W10': '15', 'W11': '18', 'W2': '16', 'W3': '18', 'W4': '20', 'W5': '17', 'W6': '19', 'W7': '16', 'W8': '18', 'W9': '15'}}, {'data_row_number': 8, 'values': {'Unnamed: 0': 'W9', 'W1': '17', 'W10': '15', 'W11': '18', 'W2': '18', 'W3': '12', 'W4': '14', 'W5': '16', 'W6': '15', 'W7': '14', 'W8': '17', 'W9': '21'}}, {'data_row_number': 9, 'values': {'Unnamed: 0': 'W10', 'W1': '14', 'W10': '17', 'W11': '19', 'W2': '13', 'W3': '15', 'W4': '17', 'W5': '16', 'W6': '18', 'W7': '14', 'W8': '19', 'W9': '15'}}, {'data_row_number': 10, 'values': {'Unnamed: 0': 'W11', 'W1': '15', 'W10': '21', 'W11': '13', 'W2': '13', 'W3': '16', 'W4': '17', 'W5': '11', 'W6': '13', 'W7': '14', 'W8': '15', 'W9': '19'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB
tables = CSVQA_DATA['tables']

def get_table(table_id):
    for t in tables:
        if t['table_id'] == table_id:
            return t
    raise ValueError(f'Table {table_id} not found')
ware_table = get_table('file_0_view_0')
ware_ids = []
capacity = {}
opening_cost = {}
for rec in ware_table['records']:
    vals = rec['values']
    wid = vals['Warehouse (i)'].strip()
    ware_ids.append(wid)
    capacity[wid] = float(vals['Capacity (units)'])
    opening_cost[wid] = float(vals['Opening Cost (fi)'])
store_table = get_table('file_1_view_0')
store_ids = []
demand = {}
for rec in store_table['records']:
    vals = rec['values']
    sid = vals['Store (j)'].strip()
    store_ids.append(sid)
    demand[sid] = float(vals['Demand (units, dj)'])
trans_table = get_table('file_2_view_0')
col_names = [c for c in trans_table['columns'] if c != 'Unnamed: 0']
cost = {}
for rec in trans_table['records']:
    vals = rec['values']
    row_label = vals['Unnamed: 0'].strip()
    wid = row_label.lstrip('W')
    cost[wid] = {}
    for col in col_names:
        sid = col.lstrip('W')
        cost[wid][sid] = float(vals[col])
for wid in ware_ids:
    if wid not in capacity or wid not in opening_cost:
        raise ValueError(f'Missing capacity or opening cost for warehouse {wid}')
    if wid not in cost:
        raise ValueError(f'Missing transportation costs for warehouse {wid}')
    for sid in store_ids:
        if sid not in cost[wid]:
            raise ValueError(f'Missing cost from warehouse {wid} to store {sid}')
for sid in store_ids:
    if sid not in demand:
        raise ValueError(f'Missing demand for store {sid}')

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(ware_ids, lb=0, ub=1, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in ware_ids for j in store_ids], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((opening_cost[i] * y[i] for i in ware_ids)) + gurobipy.quicksum((cost[i][j] * x[i, j] for i in ware_ids for j in store_ids))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in ware_ids)) >= demand[j] for j in store_ids), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in store_ids)) <= capacity[i] * y[i] for i in ware_ids), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
