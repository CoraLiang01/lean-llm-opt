CSVQA_DATA = {'query': "On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which warehouses should be activated so that the demand of all musicians and bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the total supply obtained from different warehouses.", 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1083'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '776'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '16214'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    demand_table = tables['file_0_view_0']
    d_j = {}
    for rec in demand_table['records']:
        vals = rec['values']
        cust = vals['customer']
        d_j[cust] = float(vals['demand'])
    fixed_table = tables['file_1_view_0']
    f_i = {}
    for rec in fixed_table['records']:
        vals = rec['values']
        wh = vals['Unnamed: 0']
        f_i[wh] = float(vals['fixed_costs'])
    trans_table = tables['file_2_view_0']
    c_ij = {}
    for rec in trans_table['records']:
        vals = rec['values']
        wh = vals['Unnamed: 0']
        for cust in d_j:
            if cust not in vals:
                raise ValueError(f'Missing cost for warehouse {wh} and customer {cust}')
            c_ij[wh, cust] = float(vals[cust])
    if set(f_i.keys()) != set((rec['values']['Unnamed: 0'] for rec in trans_table['records'])):
        raise ValueError('Warehouse IDs mismatch between fixed cost and transportation cost tables')
    if set(d_j.keys()) != set((rec['values']['customer'] for rec in demand_table['records'])):
        raise ValueError('Customer IDs mismatch between demand and transportation cost tables')
    for i in f_i:
        for j in d_j:
            if (i, j) not in c_ij:
                raise ValueError(f'Missing transportation cost for warehouse {i} and customer {j}')
    M = sum(d_j.values())
    m = gurobipy.Model('facility_location')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(f_i.keys(), vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in f_i.keys() for j in d_j.keys()], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f_i[i] * y[i] for i in f_i.keys())) + gurobipy.quicksum((c_ij[i, j] * x[i, j] for i in f_i.keys() for j in d_j.keys()))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in f_i.keys())) >= d_j[j] for j in d_j.keys()), name='')
    m.addConstrs((x[i, j] <= M * y[i] for i in f_i.keys() for j in d_j.keys()), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
