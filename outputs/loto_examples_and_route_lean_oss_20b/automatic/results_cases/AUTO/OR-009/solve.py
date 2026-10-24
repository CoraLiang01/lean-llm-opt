CSVQA_DATA = {'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is to determine the optimal quantity of beverages to be shipped from each production plant to each retail outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while minimizing the total transportation cost.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '94'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '39'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '65'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '435'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '543.756480860856', 'C2': '23.685276141764653', 'C3': '23.676386730773032', 'C4': '447.75143678673766', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '883.9151090405642', 'C2': '0.04977684765576961', 'C3': '0.0350986687216299', 'C4': '44.45588531711622', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '537.3456896658107', 'C2': '23.769274659075112', 'C3': '498.95659249465467', 'C4': '440.60737890439776', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1791.493192397229', 'C2': '68.21633865655126', 'C3': '1432.4837339656747', 'C4': '1527.7635425462734', 'Unnamed: 0': 'S4'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    cust_table = tables['file_0_view_0']
    customers = []
    demand = {}
    for rec in cust_table['records']:
        cust_id = rec['values']['customer']
        customers.append(cust_id)
        try:
            demand_val = float(rec['values']['demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Invalid demand value for customer {cust_id}')
        demand[cust_id] = demand_val
    plant_table = tables['file_1_view_0']
    plants = []
    supply = {}
    for rec in plant_table['records']:
        plant_id = rec['values']['Unnamed: 0']
        plants.append(plant_id)
        try:
            supply_val = float(rec['values']['supply_capacity'])
        except (KeyError, ValueError):
            raise ValueError(f'Invalid supply capacity for plant {plant_id}')
        supply[plant_id] = supply_val
    cost_table = tables['file_2_view_0']
    cost = {}
    for rec in cost_table['records']:
        plant_id = rec['values']['Unnamed: 0']
        for cust_id in customers:
            col_name = cust_id
            if col_name not in rec['values']:
                raise ValueError(f'Missing cost for plant {plant_id} to customer {cust_id}')
            try:
                cost_val = float(rec['values'][col_name])
            except ValueError:
                raise ValueError(f'Invalid cost value for plant {plant_id} to customer {cust_id}')
            cost[plant_id, cust_id] = cost_val
    for s in plants:
        if s not in supply:
            raise ValueError(f'Supply capacity missing for plant {s}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Demand missing for customer {c}')
    for s in plants:
        for c in customers:
            if (s, c) not in cost:
                raise ValueError(f'Cost missing for plant {s} to customer {c}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(s, c) for s in plants for c in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj_expr = gurobipy.quicksum((cost[s, c] * x[s, c] for s in plants for c in customers))
    model.setObjective(obj_expr, GRB.MINIMIZE)
    for c in customers:
        expr = gurobipy.quicksum((x[s, c] for s in plants))
        model.addConstr(expr == demand[c], name=f'demand_{c}')
    for s in plants:
        expr = gurobipy.quicksum((x[s, c] for c in customers))
        model.addConstr(expr <= supply[s], name=f'capacity_{s}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
