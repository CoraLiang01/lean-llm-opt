CSVQA_DATA = {'query': "In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting operations, with the fixed cost data provided in the “fixed_cost.csv” file. Each branch needs to source a certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of goods from each supplier is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all branches is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply obtained from different suppliers. These decision variables help determine the optimal allocation of supply to minimize the total of fixed and transportation costs.", 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '143'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '6'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '10'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '25'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '3'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '150.74', 'C2': '0.02', 'C3': '49.13', 'C4': '2080.15', 'C5': '426.4', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '233.05', 'C2': '97.73', 'C3': '49.84', 'C4': '1982.39', 'C5': '23.96', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '55.68', 'C2': '935.61', 'C3': '4.03', 'C4': '73.09', 'C5': '525.3200000000001', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1483.82', 'C2': '1801.08', 'C3': '112.16', 'C4': '816.05', 'C5': '107.01', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1119.47', 'C2': '884.3099999999999', 'C3': '0.08', 'C4': '1544.95', 'C5': '543.67', 'Unnamed: 0': 'S5'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    demand_table = tables['file_0_view_0']
    J = []
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer']
        J.append(cust)
        demand[cust] = float(rec['values']['demand'])
    fixed_table = tables['file_1_view_0']
    I = []
    fixed_cost = {}
    for rec in fixed_table['records']:
        sup = rec['values']['Unnamed: 0']
        I.append(sup)
        fixed_cost[sup] = float(rec['values']['fixed_costs'])
    trans_table = tables['file_2_view_0']
    for sup in I:
        if sup not in [rec['values']['Unnamed: 0'] for rec in trans_table['records']]:
            raise ValueError(f'Supplier {sup} missing in transportation table')
    for cust in J:
        if cust not in trans_table['columns'][1:]:
            raise ValueError(f'Customer {cust} missing in transportation columns')
    c = {}
    for rec in trans_table['records']:
        sup = rec['values']['Unnamed: 0']
        for cust in J:
            c[sup, cust] = float(rec['values'][cust])
    M = sum(demand.values())
    m = gurobipy.Model('SupplierCustomer')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in I)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) >= demand[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()
