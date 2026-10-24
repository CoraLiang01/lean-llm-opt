CSVQA_DATA = {'query': 'A retail company wants to decide where to open warehouses to minimize total costs while meeting the demands of several stores in a region. Each potential warehouse has an opening cost, and there is a known transportation cost for supplying goods to each store from each warehouse. Each warehouse has a capacity, and each store has a specific demand. The potential warehouses costs are in PotentialWarehouses_Costs.csv, demand for each store is in Stores_Demands.csv and the transportation costs c_ij from warehouse i to store j is in TransportationCost.csv\n\n    You are to determine which warehouses to open and how to assign the demand of each store to the warehouses such that all store demands are met, warehouse capacities are not exceeded, and the total cost (opening + transportation) is minimized.', 'relationships': [{'column_axis': {'id_column': 'Warehouse (i)', 'table_id': 'file_0_view_0'}, 'column_id_mapping': {'W1': '1', 'W10': '10', 'W11': '11', 'W2': '2', 'W3': '3', 'W4': '4', 'W5': '5', 'W6': '6', 'W7': '7', 'W8': '8', 'W9': '9'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Store (j)', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'row_id_mapping': {'W1': '1', 'W10': '10', 'W11': '11', 'W2': '2', 'W3': '3', 'W4': '4', 'W5': '5', 'W6': '6', 'W7': '7', 'W8': '8', 'W9': '9'}, 'type': 'matrix'}], 'tables': [{'columns': ['Warehouse (i)', 'Opening Cost (fi)', 'Capacity (units)'], 'file_name': 'PotentialWarehouses_Costs.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '3000', 'Warehouse (i)': '1'}}, {'data_row_number': 1, 'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '3200', 'Warehouse (i)': '2'}}, {'data_row_number': 2, 'values': {'Capacity (units)': '200', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '3'}}, {'data_row_number': 3, 'values': {'Capacity (units)': '150', 'Opening Cost (fi)': '2800', 'Warehouse (i)': '4'}}, {'data_row_number': 4, 'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3500', 'Warehouse (i)': '5'}}, {'data_row_number': 5, 'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2700', 'Warehouse (i)': '6'}}, {'data_row_number': 6, 'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '2900', 'Warehouse (i)': '7'}}, {'data_row_number': 7, 'values': {'Capacity (units)': '175', 'Opening Cost (fi)': '3050', 'Warehouse (i)': '8'}}, {'data_row_number': 8, 'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '9'}}, {'data_row_number': 9, 'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '2200', 'Warehouse (i)': '10'}}, {'data_row_number': 10, 'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2890', 'Warehouse (i)': '11'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Store (j)', 'Demand (units, dj)'], 'file_name': 'Stores_Demands.csv', 'records': [{'data_row_number': 0, 'values': {'Demand (units, dj)': '30', 'Store (j)': '1'}}, {'data_row_number': 1, 'values': {'Demand (units, dj)': '40', 'Store (j)': '2'}}, {'data_row_number': 2, 'values': {'Demand (units, dj)': '20', 'Store (j)': '3'}}, {'data_row_number': 3, 'values': {'Demand (units, dj)': '35', 'Store (j)': '4'}}, {'data_row_number': 4, 'values': {'Demand (units, dj)': '20', 'Store (j)': '5'}}, {'data_row_number': 5, 'values': {'Demand (units, dj)': '25', 'Store (j)': '6'}}, {'data_row_number': 6, 'values': {'Demand (units, dj)': '45', 'Store (j)': '7'}}, {'data_row_number': 7, 'values': {'Demand (units, dj)': '38', 'Store (j)': '8'}}, {'data_row_number': 8, 'values': {'Demand (units, dj)': '32', 'Store (j)': '9'}}, {'data_row_number': 9, 'values': {'Demand (units, dj)': '41', 'Store (j)': '10'}}, {'data_row_number': 10, 'values': {'Demand (units, dj)': '44', 'Store (j)': '11'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11'], 'file_name': 'TransportationCost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'W1', 'W1': '12', 'W10': '14', 'W11': '15', 'W2': '11', 'W3': '14', 'W4': '15', 'W5': '17', 'W6': '13', 'W7': '12', 'W8': '16', 'W9': '16'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'W2', 'W1': '17', 'W10': '15', 'W11': '16', 'W2': '19', 'W3': '15', 'W4': '20', 'W5': '18', 'W6': '14', 'W7': '17', 'W8': '15', 'W9': '13'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'W3', 'W1': '13', 'W10': '18', 'W11': '17', 'W2': '14', 'W3': '12', 'W4': '14', 'W5': '16', 'W6': '15', 'W7': '11', 'W8': '14', 'W9': '16'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'W4', 'W1': '18', 'W10': '13', 'W11': '18', 'W2': '16', 'W3': '17', 'W4': '13', 'W5': '18', 'W6': '17', 'W7': '14', 'W8': '19', 'W9': '16'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'W5', 'W1': '10', 'W10': '15', 'W11': '17', 'W2': '13', 'W3': '12', 'W4': '19', 'W5': '15', 'W6': '11', 'W7': '12', 'W8': '14', 'W9': '12'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'W6', 'W1': '15', 'W10': '18', 'W11': '19', 'W2': '12', 'W3': '14', 'W4': '16', 'W5': '13', 'W6': '17', 'W7': '16', 'W8': '16', 'W9': '14'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'W7', 'W1': '14', 'W10': '16', 'W11': '14', 'W2': '13', 'W3': '15', 'W4': '17', 'W5': '12', 'W6': '13', 'W7': '14', 'W8': '15', 'W9': '12'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'W8', 'W1': '19', 'W10': '15', 'W11': '18', 'W2': '16', 'W3': '18', 'W4': '20', 'W5': '17', 'W6': '19', 'W7': '16', 'W8': '18', 'W9': '15'}}, {'data_row_number': 8, 'values': {'Unnamed: 0': 'W9', 'W1': '17', 'W10': '15', 'W11': '18', 'W2': '18', 'W3': '12', 'W4': '14', 'W5': '16', 'W6': '15', 'W7': '14', 'W8': '17', 'W9': '21'}}, {'data_row_number': 9, 'values': {'Unnamed: 0': 'W10', 'W1': '14', 'W10': '17', 'W11': '19', 'W2': '13', 'W3': '15', 'W4': '17', 'W5': '16', 'W6': '18', 'W7': '14', 'W8': '19', 'W9': '15'}}, {'data_row_number': 10, 'values': {'Unnamed: 0': 'W11', 'W1': '15', 'W10': '21', 'W11': '13', 'W2': '13', 'W3': '16', 'W4': '17', 'W5': '11', 'W6': '13', 'W7': '14', 'W8': '15', 'W9': '19'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def get_table(table_id):
    for tbl in CSVQA_DATA['tables']:
        if tbl['table_id'] == table_id:
            return tbl
    raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
ware_tbl = get_table('file_0_view_0')
ware_ids = []
f = {}
C = {}
for rec in ware_tbl['records']:
    wid = rec['values']['Warehouse (i)'].strip()
    ware_ids.append(wid)
    f[wid] = float(rec['values']['Opening Cost (fi)'])
    C[wid] = float(rec['values']['Capacity (units)'])
store_tbl = get_table('file_1_view_0')
store_ids = []
d = {}
for rec in store_tbl['records']:
    sid = rec['values']['Store (j)'].strip()
    store_ids.append(sid)
    d[sid] = float(rec['values']['Demand (units, dj)'])
mat_tbl = get_table('file_2_view_0')
rel = None
for r in CSVQA_DATA['relationships']:
    if r.get('matrix_table_id') == 'file_2_view_0':
        rel = r
        break
if rel is None:
    raise ValueError('Relationship for matrix table not found')
row_id_mapping = rel.get('row_id_mapping', {})
col_id_mapping = rel.get('column_id_mapping', {})
c = {}
for rec in mat_tbl['records']:
    raw_row = rec['values']['Unnamed: 0'].strip()
    if raw_row not in row_id_mapping:
        raise ValueError(f'Row id mapping missing for {raw_row}')
    sid = row_id_mapping[raw_row]
    if sid not in store_ids:
        raise ValueError(f'Store id {sid} from matrix not in store list')
    for raw_col, val in rec['values'].items():
        if raw_col == 'Unnamed: 0':
            continue
        if raw_col not in col_id_mapping:
            raise ValueError(f'Column id mapping missing for {raw_col}')
        wid = col_id_mapping[raw_col]
        if wid not in ware_ids:
            raise ValueError(f'Warehouse id {wid} from matrix not in warehouse list')
        c[wid, sid] = float(val)
for wid in ware_ids:
    if wid not in f or wid not in C:
        raise ValueError(f'Missing parameters for warehouse {wid}')
for sid in store_ids:
    if sid not in d:
        raise ValueError(f'Missing demand for store {sid}')
for wid in ware_ids:
    for sid in store_ids:
        if (wid, sid) not in c:
            raise ValueError(f'Missing transportation cost for warehouse {wid} to store {sid}')
m = gurobipy.Model('Warehouse_Opening')
m.setParam('MIPGap', 1e-06)
y = m.addVars(ware_ids, vtype=GRB.BINARY, name='')
x = m.addVars([(wid, sid) for wid in ware_ids for sid in store_ids], vtype=GRB.INTEGER, lb=0, name='')
obj = gurobipy.quicksum((f[wid] * y[wid] for wid in ware_ids)) + gurobipy.quicksum((c[wid, sid] * x[wid, sid] for wid in ware_ids for sid in store_ids))
m.setObjective(obj, GRB.MINIMIZE)
for sid in store_ids:
    m.addConstr(gurobipy.quicksum((x[wid, sid] for wid in ware_ids)) == d[sid], name=f'demand_{sid}')
for wid in ware_ids:
    m.addConstr(gurobipy.quicksum((x[wid, sid] for sid in store_ids)) <= C[wid] * y[wid], name=f'capacity_{wid}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
