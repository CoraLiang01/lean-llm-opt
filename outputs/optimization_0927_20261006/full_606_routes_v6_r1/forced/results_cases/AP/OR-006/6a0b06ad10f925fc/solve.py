import gurobipy as gp
from gurobipy import GRB
warehouses = [{'id': 'S1', 'supply_capacity': 127}, {'id': 'S2', 'supply_capacity': 236}, {'id': 'S3', 'supply_capacity': 168}, {'id': 'S4', 'supply_capacity': 115}, {'id': 'S5', 'supply_capacity': 280}, {'id': 'S6', 'supply_capacity': 179}, {'id': 'S7', 'supply_capacity': 135}, {'id': 'S8', 'supply_capacity': 263}, {'id': 'S9', 'supply_capacity': 283}, {'id': 'S10', 'supply_capacity': 476}]
customers = [{'id': 'C1', 'demand': 45}, {'id': 'C2', 'demand': 23}, {'id': 'C3', 'demand': 94}, {'id': 'C4', 'demand': 92}, {'id': 'C5', 'demand': 57}, {'id': 'C6', 'demand': 52}, {'id': 'C7', 'demand': 23}, {'id': 'C8', 'demand': 99}, {'id': 'C9', 'demand': 99}, {'id': 'C10', 'demand': 77}]
transportation_costs = {'S1': {'C1': 2077.058672521021, 'C2': 0.0, 'C3': 54.33526480458508, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629162332, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33026926588778, 'C10': 0.0}, 'S2': {'C1': 2077.058672521021, 'C2': 0.0, 'C3': 1141.0405608962865, 'C4': 0.0, 'C5': 0.0, 'C6': 651.1112332492198, 'C7': 0.0, 'C8': 0.0, 'C9': 8.063346155518467, 'C10': 0.0}, 'S3': {'C1': 79.9210295982608, 'C2': 474.24509131006675, 'C3': 1477.0676289106607, 'C4': 22.583099586193658, 'C5': 474.24509131006675, 'C6': 41.106596962251096, 'C7': 474.24509131006675, 'C8': 474.24509131006675, 'C9': 624.162539502301, 'C10': 474.24509131006675}, 'S4': {'C1': 1659.336929105112, 'C2': 57.20541468776147, 'C3': 186.1519048103841, 'C4': 1201.3137084429907, 'C5': 1029.6974643797064, 'C6': 41.82210594495074, 'C7': 57.20541468776147, 'C8': 1201.3137084429907, 'C9': 884.5633870657458, 'C10': 1029.6974643797064}, 'S5': {'C1': 1297.2567040858307, 'C2': 77.76629131320436, 'C3': 24.26760227579214, 'C4': 1399.7932436376784, 'C5': 77.76629131320436, 'C6': 53.91161728496604, 'C7': 1399.7932436376784, 'C8': 77.76629131320436, 'C9': 1255.115148013589, 'C10': 1399.7932436376784}, 'S6': {'C1': 1998.9090658724567, 'C2': 985.3165435695341, 'C3': 2.8541686885814643, 'C4': 1149.53596749779, 'C5': 985.3165435695341, 'C6': 730.6923647662475, 'C7': 54.73980797608523, 'C8': 985.3165435695341, 'C9': 46.803102206463265, 'C10': 1149.53596749779}, 'S7': {'C1': 1780.3360050180179, 'C2': 0.0, 'C3': 1141.0405608962865, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629162332, 'C7': 0.0, 'C8': 0.0, 'C9': 8.063346155518467, 'C10': 0.0}, 'S8': {'C1': 75.40935896233042, 'C2': 1338.1987290721909, 'C3': 21.391345987195333, 'C4': 74.34437383734394, 'C5': 74.34437383734394, 'C6': 937.3506239061463, 'C7': 1338.1987290721909, 'C8': 1338.1987290721909, 'C9': 1392.1186581383768, 'C10': 1338.1987290721909}, 'S9': {'C1': 98.90755583433433, 'C2': 0.0, 'C3': 978.0347664825314, 'C4': 0.0, 'C5': 0.0, 'C6': 651.1112332492198, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33026926588778, 'C10': 0.0}, 'S10': {'C1': 2077.058672521021, 'C2': 0.0, 'C3': 54.33526480458508, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629162332, 'C7': 0.0, 'C8': 0.0, 'C9': 145.1402307993324, 'C10': 0.0}}
warehouse_ids = [w['id'] for w in warehouses]
customer_ids = [c['id'] for c in customers]
supply_capacity = {w['id']: w['supply_capacity'] for w in warehouses}
demand = {c['id']: c['demand'] for c in customers}
for wid in warehouse_ids:
    if wid not in transportation_costs:
        raise ValueError(f'Missing transportation_costs for warehouse {wid}')
    for cid in customer_ids:
        if cid not in transportation_costs[wid]:
            raise ValueError(f'Missing transportation_costs for warehouse {wid}, customer {cid}')
eligible_pairs = [(wid, cid) for wid in warehouse_ids for cid in customer_ids if transportation_costs[wid][cid] != 0.0]
zero_cost_pairs = [(wid, cid) for wid in warehouse_ids for cid in customer_ids if transportation_costs[wid][cid] == 0.0]
m = gp.Model('Logistics_Transportation')
x_vars = m.addVars(warehouse_ids, customer_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
for (wid, cid) in zero_cost_pairs:
    x_vars[wid, cid].lb = 0
    x_vars[wid, cid].ub = 0
m.setObjective(gp.quicksum((transportation_costs[wid][cid] * x_vars[wid, cid] for wid in warehouse_ids for cid in customer_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wid, cid] for wid in warehouse_ids)) == demand[cid] for cid in customer_ids), name='')
m.addConstrs((gp.quicksum((x_vars[wid, cid] for cid in customer_ids)) <= supply_capacity[wid] for wid in warehouse_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')