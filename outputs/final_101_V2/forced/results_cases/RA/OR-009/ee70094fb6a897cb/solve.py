LEGACY_OBSERVATION = 'customer_demand.csv\n{"values": {"customer": "C1", "demand": "94"}}\n{"values": {"customer": "C2", "demand": "39"}}\n{"values": {"customer": "C3", "demand": "65"}}\n{"values": {"customer": "C4", "demand": "435"}}\n\nsupply_capacity.csv\n{"values": {"Unnamed: 0": "S1", "supply_capacity": "2531"}}\n{"values": {"Unnamed: 0": "S2", "supply_capacity": "20"}}\n{"values": {"Unnamed: 0": "S3", "supply_capacity": "210"}}\n{"values": {"Unnamed: 0": "S4", "supply_capacity": "241"}}\n\ntransportation_costs.csv\n{"values": {"Unnamed: 0": "S1", "C1": "543.756480860856", "C2": "23.685276141764653", "C3": "23.676386730773032", "C4": "447.75143678673766"}}\n{"values": {"Unnamed: 0": "S2", "C1": "883.9151090405642", "C2": "0.04977684765576961", "C3": "0.0350986687216299", "C4": "44.45588531711622"}}\n{"values": {"Unnamed: 0": "S3", "C1": "537.3456896658107", "C2": "23.769274659075112", "C3": "498.95659249465467", "C4": "440.60737890439776"}}\n{"values": {"Unnamed: 0": "S4", "C1": "1791.493192397229", "C2": "68.21633865655126", "C3": "1432.4837339656747", "C4": "1527.7635425462734"}}'
LEGACY_RECORDS = [{'source': 'customer_demand.csv', 'values': {'customer': 'C1', 'demand': '94'}}, {'source': 'customer_demand.csv', 'values': {'customer': 'C2', 'demand': '39'}}, {'source': 'customer_demand.csv', 'values': {'customer': 'C3', 'demand': '65'}}, {'source': 'customer_demand.csv', 'values': {'customer': 'C4', 'demand': '435'}}, {'source': 'supply_capacity.csv', 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}}, {'source': 'supply_capacity.csv', 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}}, {'source': 'supply_capacity.csv', 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}}, {'source': 'supply_capacity.csv', 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '543.756480860856', 'C2': '23.685276141764653', 'C3': '23.676386730773032', 'C4': '447.75143678673766'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '883.9151090405642', 'C2': '0.04977684765576961', 'C3': '0.0350986687216299', 'C4': '44.45588531711622'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '537.3456896658107', 'C2': '23.769274659075112', 'C3': '498.95659249465467', 'C4': '440.60737890439776'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '1791.493192397229', 'C2': '68.21633865655126', 'C3': '1432.4837339656747', 'C4': '1527.7635425462734'}}]
import gurobipy as gp
from gurobipy import GRB
plants = []
customers = []
supply_capacity = {}
demand = {}
cost = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'supply_capacity.csv':
        plant = rec['values']['Unnamed: 0']
        plants.append(plant)
        supply_capacity[plant] = float(rec['values']['supply_capacity'])
    elif rec['source'] == 'customer_demand.csv':
        customer = rec['values']['customer']
        customers.append(customer)
        demand[customer] = float(rec['values']['demand'])
    elif rec['source'] == 'transportation_costs.csv':
        plant = rec['values']['Unnamed: 0']
        if plant not in cost:
            cost[plant] = {}
        for cust in rec['values']:
            if cust != 'Unnamed: 0':
                cost[plant][cust] = float(rec['values'][cust])
plants = list(dict.fromkeys(plants))
customers = list(dict.fromkeys(customers))
for i in plants:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for plant {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for plant {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for plant {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model('brewco_transport')
x = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in plants), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in plants)) >= demand[j] for j in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')