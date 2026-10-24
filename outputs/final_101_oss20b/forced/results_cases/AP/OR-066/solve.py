CSVQA_DATA = {'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with these costs detailed in “fixed_cost.csv.” Every supermarket must procure a unit of goods from the available suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in “transportation_costs.csv,” while demand data is available in “demand.csv.” The goal is to identify which suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. Binary decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the share of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the most cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '144'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB
tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
demand_table = tables['file_0_view_0']
customers = [rec['values']['customer'] for rec in demand_table['records']]
demand = {rec['values']['customer']: float(rec['values']['demand']) for rec in demand_table['records']}
fixed_table = tables['file_1_view_0']
suppliers = [rec['values']['Unnamed: 0'] for rec in fixed_table['records']]
fixed_cost = {rec['values']['Unnamed: 0']: float(rec['values']['fixed_costs']) for rec in fixed_table['records']}
transport_table = tables['file_2_view_0']
transport_cost = {}
for rec in transport_table['records']:
    supplier = rec['values']['Unnamed: 0']
    for customer in customers:
        key = (supplier, customer)
        if customer not in rec['values']:
            raise ValueError(f'Missing transportation cost for supplier {supplier} to customer {customer}')
        transport_cost[key] = float(rec['values'][customer])
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in suppliers:
    for j in customers:
        if (i, j) not in transport_cost:
            raise ValueError(f'Missing transportation cost for supplier {i} to customer {j}')

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gurobipy.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    M = max(demand.values())
    for i in suppliers:
        for j in customers:
            m.addConstr(x[i, j] <= y[i] * M, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
