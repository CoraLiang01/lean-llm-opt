CSVQA_DATA = {'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with these costs detailed in “fixed_cost.csv.” Every supermarket must procure a unit of goods from the available suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in “transportation_costs.csv,” while demand data is available in “demand.csv.” The goal is to identify which suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. Binary decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the share of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the most cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '144'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def _get_table(table_id):
    for tbl in CSVQA_DATA['tables']:
        if tbl['table_id'] == table_id:
            return tbl
    raise KeyError(f'Table {table_id} not found in CSVQA_DATA')
fixed_cost_tbl = _get_table('file_1_view_0')
I = [rec['values']['Unnamed: 0'] for rec in fixed_cost_tbl['records']]
demand_tbl = _get_table('file_0_view_0')
J = [rec['values']['customer'] for rec in demand_tbl['records']]
f = {}
for rec in fixed_cost_tbl['records']:
    key = rec['values']['Unnamed: 0']
    val = float(rec['values']['fixed_costs'])
    f[key] = val
d = {}
for rec in demand_tbl['records']:
    key = rec['values']['customer']
    val = float(rec['values']['demand'])
    d[key] = val
cost_tbl = _get_table('file_2_view_0')
c = {}
for rec in cost_tbl['records']:
    supplier = rec['values']['Unnamed: 0']
    for col in cost_tbl['columns']:
        if col == 'Unnamed: 0':
            continue
        customer = col
        val = float(rec['values'][customer])
        c[supplier, customer] = val
for i in I:
    if i not in f:
        raise ValueError(f'Missing fixed cost for supplier {i}')
for j in J:
    if j not in d:
        raise ValueError(f'Missing demand for supermarket {j}')
for i in I:
    for j in J:
        if (i, j) not in c:
            raise ValueError(f'Missing transportation cost for supplier {i} to supermarket {j}')
M = sum((d[j] for j in J))
m = gurobipy.Model('Supplier_Activation_FLP')
m.setParam('MIPGap', 1e-06)
y = m.addVars(I, vtype=GRB.BINARY, name='')
x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
obj = gurobipy.quicksum((c[i, j] * x[i, j] for i in I for j in J)) + gurobipy.quicksum((f[i] * y[i] for i in I))
m.setObjective(obj, GRB.MINIMIZE)
m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) == d[j] for j in J), name='')
m.addConstrs((gurobipy.quicksum((x[i, j] for j in J)) <= M * y[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
