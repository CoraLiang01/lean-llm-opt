SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D1', 'demand': '428'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D2', 'demand': '217'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D3', 'demand': '214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D4', 'demand': '380'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D5', 'demand': '254'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S1', 'supply_capacity': '428'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S2', 'supply_capacity': '217'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S3', 'supply_capacity': '214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S4', 'supply_capacity': '380'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S5', 'supply_capacity': '254'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'D1': '269.3910588020795', 'D2': '1.4537335390933939', 'D3': '99.60345345756605', 'D4': '26.64078166309837', 'D5': '9.537688956880922'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'D1': '9.291846876785183', 'D2': '10.874778437070223', 'D3': '144.52609291614627', 'D4': '11.420133077898234', 'D5': '153.1756819927813'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'D1': '9.674584301671008', 'D2': '2.6191650959687944', 'D3': '100.8242249168735', 'D4': '3.2121910887916876', 'D5': '133.8493396124168'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'D1': '270.57498480010247', 'D2': '32.50253586', 'D3': '4.6842098096469815', 'D4': '1.5682269686546804', 'D5': '9.58927599'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'D1': '226.0331910675782', 'D2': '8.669161980826471', 'D3': '65.47681316968448', 'D4': '9.068765258459958', 'D5': '202.65015316425533'}}]
import gurobipy as gp
from gurobipy import GRB
SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D1', 'demand': '428'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D2', 'demand': '217'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D3', 'demand': '214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D4', 'demand': '380'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'values': {'customer': 'D5', 'demand': '254'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S1', 'supply_capacity': '428'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S2', 'supply_capacity': '217'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S3', 'supply_capacity': '214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S4', 'supply_capacity': '380'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'values': {'region': 'S5', 'supply_capacity': '254'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'D1': '269.3910588020795', 'D2': '1.4537335390933939', 'D3': '99.60345345756605', 'D4': '26.64078166309837', 'D5': '9.537688956880922'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'D1': '9.291846876785183', 'D2': '10.874778437070223', 'D3': '144.52609291614627', 'D4': '11.420133077898234', 'D5': '153.1756819927813'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'D1': '9.674584301671008', 'D2': '2.6191650959687944', 'D3': '100.8242249168735', 'D4': '3.2121910887916876', 'D5': '133.8493396124168'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'D1': '270.57498480010247', 'D2': '32.50253586', 'D3': '4.6842098096469815', 'D4': '1.5682269686546804', 'D5': '9.58927599'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'D1': '226.0331910675782', 'D2': '8.669161980826471', 'D3': '65.47681316968448', 'D4': '9.068765258459958', 'D5': '202.65015316425533'}}]
warehouses = []
stores = []
demand = {}
supply_capacity = {}
cost = {}
for rec in SOURCE_RECORDS:
    if rec['source'].endswith('customer_demand.csv'):
        cust = rec['values']['customer']
        if cust not in stores:
            stores.append(cust)
        demand[cust] = float(rec['values']['demand'])
    elif rec['source'].endswith('supply_capacity.csv'):
        wh = rec['values']['region']
        if wh not in warehouses:
            warehouses.append(wh)
        supply_capacity[wh] = float(rec['values']['supply_capacity'])
    elif rec['source'].endswith('transportation_costs.csv'):
        wh = rec['values']['Unnamed: 0']
        if wh not in cost:
            cost[wh] = {}
        for j in rec['values']:
            if j == 'Unnamed: 0':
                continue
            cost[wh][j] = float(rec['values'][j])
            if j not in stores:
                stores.append(j)
for wh in warehouses:
    if wh not in cost:
        raise ValueError(f'Missing cost row for warehouse {wh}')
    for st in stores:
        if st not in cost[wh]:
            raise ValueError(f'Missing cost for warehouse {wh}, store {st}')
for st in stores:
    if st not in demand:
        raise ValueError(f'Missing demand for store {st}')
for wh in warehouses:
    if wh not in supply_capacity:
        raise ValueError(f'Missing supply capacity for warehouse {wh}')
m = gp.Model('GreenMart_Transportation')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[wh][st] * x_vars[wh, st] for wh in warehouses for st in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wh, st] for wh in warehouses)) >= demand[st] for st in stores), name='')
m.addConstrs((gp.quicksum((x_vars[wh, st] for st in stores)) <= supply_capacity[wh] for wh in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')