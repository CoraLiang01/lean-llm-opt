CSVQA_DATA = {'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with these costs detailed in “fixed_cost.csv.” Every supermarket must procure a unit of goods from the available suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in “transportation_costs.csv,” while demand data is available in “demand.csv.” The goal is to identify which suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. Binary decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the share of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the most cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '144'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    J = set()
    D = {}
    for rec in demand_table['records']:
        j = rec['values']['customer']
        J.add(j)
        D[j] = float(rec['values']['demand'])
    fixed_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    I = set()
    F = {}
    for rec in fixed_table['records']:
        i = rec['values']['Unnamed: 0']
        I.add(i)
        F[i] = float(rec['values']['fixed_costs'])
    trans_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    C = {}
    for rec in trans_table['records']:
        i = rec['values']['Unnamed: 0']
        for j in J:
            C[i, j] = float(rec['values'][j])
    M = sum(D.values())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((F[i] * y[i] for i in I)) + gurobipy.quicksum((C[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == D[j], name=f'Demand_{j}')
    for i in I:
        for j in J:
            model.addConstr(x[i, j] <= M * y[i], name=f'Link_{i}_{j}')
    model.optimize()
    return model
m = solve_problem()
