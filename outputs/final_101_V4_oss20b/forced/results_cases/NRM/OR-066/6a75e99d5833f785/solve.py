CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities '
          'are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with '
          'these costs detailed in “fixed_cost.csv.” Every supermarket must procure a unit of goods from the available '
          'suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in '
          '“transportation_costs.csv,” while demand data is available in “demand.csv.” The goal is to identify which '
          'suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. Binary '
          'decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the share '
          'of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the most '
          'cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '144'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'fixed_cost.csv'],
             'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'aliases': ['file_2_view_0', 'transportation_costs.csv'],
             'columns': ['Unnamed: 0', 'C1', 'C2'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA

def get_table(identifier):
    for tbl in data['tables']:
        if tbl['table_id'] == identifier or identifier in tbl['aliases']:
            return tbl
    raise KeyError(f'Table {identifier} not found')
demand_tbl = get_table('file_0_view_0')
customers = []
demand = {}
for rec in demand_tbl['records']:
    cust = rec['values']['customer']
    customers.append(cust)
    demand[cust] = float(rec['values']['demand'])
supplier_tbl = get_table('file_1_view_0')
suppliers = []
fixed_cost = {}
for rec in supplier_tbl['records']:
    sup = rec['values']['Unnamed: 0']
    suppliers.append(sup)
    fixed_cost[sup] = float(rec['values']['fixed_costs'])
trans_tbl = get_table('file_2_view_0')
transport_cost = {}
for rec in trans_tbl['records']:
    sup = rec['values']['Unnamed: 0']
    transport_cost[sup] = {}
    for cust in customers:
        key = f'C{cust[1:]}'
        transport_cost[sup][cust] = float(rec['values'][key])
m = gp.Model('supplier_activation')
y = {i: m.addVar(vtype=GRB.BINARY) for i in suppliers}
x = {(i, j): m.addVar(vtype=GRB.CONTINUOUS) for i in suppliers for j in customers}
obj = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj, GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i])
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'Objective value: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {GRB.Status[m.Status]}')