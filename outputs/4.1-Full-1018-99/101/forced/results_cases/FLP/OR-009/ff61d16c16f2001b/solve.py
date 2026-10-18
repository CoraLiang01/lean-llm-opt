LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv", "values": {"customer": "C1", "demand": "94"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv", "values": {"customer": "C2", "demand": "39"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv", "values": {"customer": "C3", "demand": "65"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv", "values": {"customer": "C4", "demand": "435"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv", "values": {"Unnamed: 0": "S1", "supply_capacity": "2531"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv", "values": {"Unnamed: 0": "S2", "supply_capacity": "20"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv", "values": {"Unnamed: 0": "S3", "supply_capacity": "210"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv", "values": {"Unnamed: 0": "S4", "supply_capacity": "241"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv", "values": {"Unnamed: 0": "S1", "C1": "543.756480860856", "C2": "23.685276141764653", "C3": "23.676386730773032", "C4": "447.75143678673766"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv", "values": {"Unnamed: 0": "S2", "C1": "883.9151090405642", "C2": "0.04977684765576961", "C3": "0.0350986687216299", "C4": "44.45588531711622"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv", "values": {"Unnamed: 0": "S3", "C1": "537.3456896658107", "C2": "23.769274659075112", "C3": "498.95659249465467", "C4": "440.60737890439776"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv", "values": {"Unnamed: 0": "S4", "C1": "1791.493192397229", "C2": "68.21633865655126", "C3": "1432.4837339656747", "C4": "1527.7635425462734"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv', 'values': {'customer': 'C1', 'demand': '94'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv', 'values': {'customer': 'C2', 'demand': '39'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv', 'values': {'customer': 'C3', 'demand': '65'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv', 'values': {'customer': 'C4', 'demand': '435'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv', 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv', 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv', 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv', 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '543.756480860856', 'C2': '23.685276141764653', 'C3': '23.676386730773032', 'C4': '447.75143678673766'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '883.9151090405642', 'C2': '0.04977684765576961', 'C3': '0.0350986687216299', 'C4': '44.45588531711622'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '537.3456896658107', 'C2': '23.769274659075112', 'C3': '498.95659249465467', 'C4': '440.60737890439776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '1791.493192397229', 'C2': '68.21633865655126', 'C3': '1432.4837339656747', 'C4': '1527.7635425462734'}}]
import gurobipy as gp
from gurobipy import GRB
plants = []
retail_outlets = []
demand = {}
supply_capacity = {}
cost = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('customer_demand.csv'):
        cust = rec['values']['customer']
        val = float(rec['values']['demand'])
        if cust not in retail_outlets:
            retail_outlets.append(cust)
        demand[cust] = val
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('supply_capacity.csv'):
        plant = rec['values']['Unnamed: 0']
        val = float(rec['values']['supply_capacity'])
        if plant not in plants:
            plants.append(plant)
        supply_capacity[plant] = val
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('transportation_costs.csv'):
        plant = rec['values']['Unnamed: 0']
        if plant not in cost:
            cost[plant] = {}
        for cust in retail_outlets:
            if cust in rec['values']:
                cost[plant][cust] = float(rec['values'][cust])
for plant in plants:
    if plant not in cost:
        raise ValueError(f'Missing cost data for plant {plant}')
    for cust in retail_outlets:
        if cust not in cost[plant]:
            raise ValueError(f'Missing cost data for plant {plant}, customer {cust}')
for cust in retail_outlets:
    if cust not in demand:
        raise ValueError(f'Missing demand data for customer {cust}')
for plant in plants:
    if plant not in supply_capacity:
        raise ValueError(f'Missing supply capacity for plant {plant}')
m = gp.Model('BrewCo_Transportation')
x_vars = m.addVars(plants, retail_outlets, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[plant][cust] * x_vars[plant, cust] for plant in plants for cust in retail_outlets)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[plant, cust] for plant in plants)) == demand[cust] for cust in retail_outlets), name='')
m.addConstrs((gp.quicksum((x_vars[plant, cust] for cust in retail_outlets)) <= supply_capacity[plant] for plant in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')