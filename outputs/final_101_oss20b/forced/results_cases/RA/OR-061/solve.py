CSVQA_DATA = {'query': "In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting operations, with the fixed cost data provided in the “fixed_cost.csv” file. Each branch needs to source a certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of goods from each supplier is recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all branches is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply obtained from different suppliers. These decision variables help determine the optimal allocation of supply to minimize the total of fixed and transportation costs.", 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '143'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '6'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '10'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '25'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '3'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '150.74', 'C2': '0.02', 'C3': '49.13', 'C4': '2080.15', 'C5': '426.4', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '233.05', 'C2': '97.73', 'C3': '49.84', 'C4': '1982.39', 'C5': '23.96', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '55.68', 'C2': '935.61', 'C3': '4.03', 'C4': '73.09', 'C5': '525.3200000000001', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1483.82', 'C2': '1801.08', 'C3': '112.16', 'C4': '816.05', 'C5': '107.01', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1119.47', 'C2': '884.3099999999999', 'C3': '0.08', 'C4': '1544.95', 'C5': '543.67', 'Unnamed: 0': 'S5'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB
tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
demand_table = tables.get('file_0_view_0')
if demand_table is None:
    raise ValueError("Missing demand table 'file_0_view_0'")
demand_records = demand_table['records']
demand = {}
for rec in demand_records:
    cust = rec['values']['customer']
    val = float(rec['values']['demand'])
    demand[cust] = val
customers = list(demand.keys())
fixed_table = tables.get('file_1_view_0')
if fixed_table is None:
    raise ValueError("Missing fixed cost table 'file_1_view_0'")
fixed_records = fixed_table['records']
fixed_cost = {}
for rec in fixed_records:
    sup = rec['values']['Unnamed: 0']
    val = float(rec['values']['fixed_costs'])
    fixed_cost[sup] = val
suppliers = list(fixed_cost.keys())
trans_table = tables.get('file_2_view_0')
if trans_table is None:
    raise ValueError("Missing transportation cost table 'file_2_view_0'")
trans_records = trans_table['records']
c = {}
for rec in trans_records:
    sup = rec['values']['Unnamed: 0']
    if sup not in suppliers:
        raise ValueError(f'Supplier {sup} in transportation table not in fixed cost table')
    for cust in customers:
        if cust not in rec['values']:
            raise ValueError(f'Customer {cust} missing in transportation record for supplier {sup}')
        c[sup, cust] = float(rec['values'][cust])
M = sum(demand.values())
for sup in suppliers:
    for cust in customers:
        if (sup, cust) not in c:
            raise ValueError(f'Missing transportation cost for supplier {sup} and customer {cust}')

def solve_problem():
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in suppliers for j in customers))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) == demand[j])
    for i in suppliers:
        for j in customers:
            model.addConstr(x[i, j] <= M * y[i])
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
