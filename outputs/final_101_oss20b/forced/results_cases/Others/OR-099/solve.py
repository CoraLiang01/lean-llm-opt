CSVQA_DATA = {'query': 'A retail company wants to decide where to open warehouses to minimize total costs while meeting the demands of several stores in a region. Each potential warehouse has an opening cost, and there is a known transportation cost for supplying goods to each store from each warehouse. Each warehouse has a capacity, and each store has a specific demand. The potential warehouses costs are in PotentialWarehouses_Costs.csv, demand for each store is in Stores_Demands.csv and the transportation costs c_ij from warehouse i to store j is in TransportationCost.csv\n\n    You are to determine which warehouses to open and how to assign the demand of each store to the warehouses such that all store demands are met, warehouse capacities are not exceeded, and the total cost (opening + transportation) is minimized.', 'relationships': [{'column_axis': {'id_column': 'Warehouse (i)', 'table_id': 'file_0_view_0'}, 'column_id_mapping': {'W1': '1', 'W10': '10', 'W11': '11', 'W2': '2', 'W3': '3', 'W4': '4', 'W5': '5', 'W6': '6', 'W7': '7', 'W8': '8', 'W9': '9'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Store (j)', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'row_id_mapping': {'W1': '1', 'W10': '10', 'W11': '11', 'W2': '2', 'W3': '3', 'W4': '4', 'W5': '5', 'W6': '6', 'W7': '7', 'W8': '8', 'W9': '9'}, 'type': 'matrix'}], 'tables': [{'columns': ['Warehouse (i)', 'Opening Cost (fi)', 'Capacity (units)'], 'file_name': 'PotentialWarehouses_Costs.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '3000', 'Warehouse (i)': '1'}}, {'data_row_number': 1, 'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '3200', 'Warehouse (i)': '2'}}, {'data_row_number': 2, 'values': {'Capacity (units)': '200', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '3'}}, {'data_row_number': 3, 'values': {'Capacity (units)': '150', 'Opening Cost (fi)': '2800', 'Warehouse (i)': '4'}}, {'data_row_number': 4, 'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3500', 'Warehouse (i)': '5'}}, {'data_row_number': 5, 'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2700', 'Warehouse (i)': '6'}}, {'data_row_number': 6, 'values': {'Capacity (units)': '160', 'Opening Cost (fi)': '2900', 'Warehouse (i)': '7'}}, {'data_row_number': 7, 'values': {'Capacity (units)': '175', 'Opening Cost (fi)': '3050', 'Warehouse (i)': '8'}}, {'data_row_number': 8, 'values': {'Capacity (units)': '170', 'Opening Cost (fi)': '3100', 'Warehouse (i)': '9'}}, {'data_row_number': 9, 'values': {'Capacity (units)': '180', 'Opening Cost (fi)': '2200', 'Warehouse (i)': '10'}}, {'data_row_number': 10, 'values': {'Capacity (units)': '190', 'Opening Cost (fi)': '2890', 'Warehouse (i)': '11'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Store (j)', 'Demand (units, dj)'], 'file_name': 'Stores_Demands.csv', 'records': [{'data_row_number': 0, 'values': {'Demand (units, dj)': '30', 'Store (j)': '1'}}, {'data_row_number': 1, 'values': {'Demand (units, dj)': '40', 'Store (j)': '2'}}, {'data_row_number': 2, 'values': {'Demand (units, dj)': '20', 'Store (j)': '3'}}, {'data_row_number': 3, 'values': {'Demand (units, dj)': '35', 'Store (j)': '4'}}, {'data_row_number': 4, 'values': {'Demand (units, dj)': '20', 'Store (j)': '5'}}, {'data_row_number': 5, 'values': {'Demand (units, dj)': '25', 'Store (j)': '6'}}, {'data_row_number': 6, 'values': {'Demand (units, dj)': '45', 'Store (j)': '7'}}, {'data_row_number': 7, 'values': {'Demand (units, dj)': '38', 'Store (j)': '8'}}, {'data_row_number': 8, 'values': {'Demand (units, dj)': '32', 'Store (j)': '9'}}, {'data_row_number': 9, 'values': {'Demand (units, dj)': '41', 'Store (j)': '10'}}, {'data_row_number': 10, 'values': {'Demand (units, dj)': '44', 'Store (j)': '11'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11'], 'file_name': 'TransportationCost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'W1', 'W1': '12', 'W10': '14', 'W11': '15', 'W2': '11', 'W3': '14', 'W4': '15', 'W5': '17', 'W6': '13', 'W7': '12', 'W8': '16', 'W9': '16'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'W2', 'W1': '17', 'W10': '15', 'W11': '16', 'W2': '19', 'W3': '15', 'W4': '20', 'W5': '18', 'W6': '14', 'W7': '17', 'W8': '15', 'W9': '13'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'W3', 'W1': '13', 'W10': '18', 'W11': '17', 'W2': '14', 'W3': '12', 'W4': '14', 'W5': '16', 'W6': '15', 'W7': '11', 'W8': '14', 'W9': '16'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'W4', 'W1': '18', 'W10': '13', 'W11': '18', 'W2': '16', 'W3': '17', 'W4': '13', 'W5': '18', 'W6': '17', 'W7': '14', 'W8': '19', 'W9': '16'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'W5', 'W1': '10', 'W10': '15', 'W11': '17', 'W2': '13', 'W3': '12', 'W4': '19', 'W5': '15', 'W6': '11', 'W7': '12', 'W8': '14', 'W9': '12'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'W6', 'W1': '15', 'W10': '18', 'W11': '19', 'W2': '12', 'W3': '14', 'W4': '16', 'W5': '13', 'W6': '17', 'W7': '16', 'W8': '16', 'W9': '14'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'W7', 'W1': '14', 'W10': '16', 'W11': '14', 'W2': '13', 'W3': '15', 'W4': '17', 'W5': '12', 'W6': '13', 'W7': '14', 'W8': '15', 'W9': '12'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'W8', 'W1': '19', 'W10': '15', 'W11': '18', 'W2': '16', 'W3': '18', 'W4': '20', 'W5': '17', 'W6': '19', 'W7': '16', 'W8': '18', 'W9': '15'}}, {'data_row_number': 8, 'values': {'Unnamed: 0': 'W9', 'W1': '17', 'W10': '15', 'W11': '18', 'W2': '18', 'W3': '12', 'W4': '14', 'W5': '16', 'W6': '15', 'W7': '14', 'W8': '17', 'W9': '21'}}, {'data_row_number': 9, 'values': {'Unnamed: 0': 'W10', 'W1': '14', 'W10': '17', 'W11': '19', 'W2': '13', 'W3': '15', 'W4': '17', 'W5': '16', 'W6': '18', 'W7': '14', 'W8': '19', 'W9': '15'}}, {'data_row_number': 10, 'values': {'Unnamed: 0': 'W11', 'W1': '15', 'W10': '21', 'W11': '13', 'W2': '13', 'W3': '16', 'W4': '17', 'W5': '11', 'W6': '13', 'W7': '14', 'W8': '15', 'W9': '19'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    wh_table = get_table('file_0_view_0')
    f_i = {}
    C_i = {}
    for rec in wh_table['records']:
        wid = int(rec['values']['Warehouse (i)'])
        f_i[wid] = float(rec['values']['Opening Cost (fi)'])
        C_i[wid] = float(rec['values']['Capacity (units)'])
    store_table = get_table('file_1_view_0')
    d_j = {}
    for rec in store_table['records']:
        sid = int(rec['values']['Store (j)'])
        d_j[sid] = float(rec['values']['Demand (units, dj)'])
    rel = None
    for r in CSVQA_DATA['relationships']:
        if r.get('type') == 'matrix':
            rel = r
            break
    if rel is None:
        raise ValueError('Matrix relationship not found')
    col_axis = rel['column_axis']
    row_axis = rel['row_axis']
    col_id_mapping = rel.get('column_id_mapping', {})
    row_id_mapping = rel.get('row_id_mapping', {})
    matrix_table = get_table(rel['matrix_table_id'])
    c_ij = {}
    for rec in matrix_table['records']:
        row_label = rec['values'][rel['row_id_column']]
        j = int(row_id_mapping.get(row_label, row_label))
        for col_label in matrix_table['columns']:
            if col_label == rel['row_id_column']:
                continue
            i = int(col_id_mapping.get(col_label, col_label))
            val = float(rec['values'][col_label])
            c_ij[i, j] = val
    I = set(f_i.keys())
    J = set(d_j.keys())
    for i in I:
        if i not in f_i or i not in C_i:
            raise ValueError(f'Missing parameters for warehouse {i}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for store {j}')
    for i in I:
        for j in J:
            if (i, j) not in c_ij:
                raise ValueError(f'Missing transportation cost for warehouse {i} to store {j}')
    model = gurobipy.Model('WarehouseLocation')
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f_i[i] * y[i] for i in I)) + gurobipy.quicksum((c_ij[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == d_j[j], name=f'demand_{j}')
    for i in I:
        model.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= C_i[i] * y[i], name=f'capacity_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(model.ObjVal)
        for var in model.getVars():
            print(var.VarName, var.X)
    else:
        print(model.Status)
    return model
m = solve_problem()
