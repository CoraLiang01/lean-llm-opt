LEGACY_OBSERVATION = 'customer_demand.csv\n{"customer": "D1", "demand": "428"}\n{"customer": "D2", "demand": "217"}\n{"customer": "D3", "demand": "214"}\n{"customer": "D4", "demand": "380"}\n{"customer": "D5", "demand": "254"}\n\nsupply_capacity.csv\n{"region": "S1", "supply_capacity": "428"}\n{"region": "S2", "supply_capacity": "217"}\n{"region": "S3", "supply_capacity": "214"}\n{"region": "S4", "supply_capacity": "380"}\n{"region": "S5", "supply_capacity": "254"}\n\ntransportation_costs.csv\n{"Unnamed: 0": "S1", "D1": "269.3910588020795", "D2": "1.4537335390933939", "D3": "99.60345345756605", "D4": "26.64078166309837", "D5": "9.537688956880922"}\n{"Unnamed: 0": "S2", "D1": "9.291846876785183", "D2": "10.874778437070223", "D3": "144.52609291614627", "D4": "11.420133077898234", "D5": "153.1756819927813"}\n{"Unnamed: 0": "S3", "D1": "9.674584301671008", "D2": "2.6191650959687944", "D3": "100.8242249168735", "D4": "3.2121910887916876", "D5": "133.8493396124168"}\n{"Unnamed: 0": "S4", "D1": "270.57498480010247", "D2": "32.50253586", "D3": "4.6842098096469815", "D4": "1.5682269686546804", "D5": "9.58927599"}\n{"Unnamed: 0": "S5", "D1": "226.0331910675782", "D2": "8.669161980826471", "D3": "65.47681316968448", "D4": "9.068765258459958", "D5": "202.65015316425533"}'
LEGACY_RECORDS = [{'source': 'customer_demand.csv', 'values': {'customer': 'D1', 'demand': '428'}}, {'source': 'customer_demand.csv', 'values': {'customer': 'D2', 'demand': '217'}}, {'source': 'customer_demand.csv', 'values': {'customer': 'D3', 'demand': '214'}}, {'source': 'customer_demand.csv', 'values': {'customer': 'D4', 'demand': '380'}}, {'source': 'customer_demand.csv', 'values': {'customer': 'D5', 'demand': '254'}}, {'source': 'supply_capacity.csv', 'values': {'region': 'S1', 'supply_capacity': '428'}}, {'source': 'supply_capacity.csv', 'values': {'region': 'S2', 'supply_capacity': '217'}}, {'source': 'supply_capacity.csv', 'values': {'region': 'S3', 'supply_capacity': '214'}}, {'source': 'supply_capacity.csv', 'values': {'region': 'S4', 'supply_capacity': '380'}}, {'source': 'supply_capacity.csv', 'values': {'region': 'S5', 'supply_capacity': '254'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'D1': '269.3910588020795', 'D2': '1.4537335390933939', 'D3': '99.60345345756605', 'D4': '26.64078166309837', 'D5': '9.537688956880922'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'D1': '9.291846876785183', 'D2': '10.874778437070223', 'D3': '144.52609291614627', 'D4': '11.420133077898234', 'D5': '153.1756819927813'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'D1': '9.674584301671008', 'D2': '2.6191650959687944', 'D3': '100.8242249168735', 'D4': '3.2121910887916876', 'D5': '133.8493396124168'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'D1': '270.57498480010247', 'D2': '32.50253586', 'D3': '4.6842098096469815', 'D4': '1.5682269686546804', 'D5': '9.58927599'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'D1': '226.0331910675782', 'D2': '8.669161980826471', 'D3': '65.47681316968448', 'D4': '9.068765258459958', 'D5': '202.65015316425533'}}]
import gurobipy as gp
from gurobipy import GRB
customer_demand = {}
supply_capacity = {}
transportation_costs = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'customer_demand.csv':
        cust = rec['values']['customer']
        demand = int(rec['values']['demand'])
        customer_demand[cust] = demand
    elif rec['source'] == 'supply_capacity.csv':
        region = rec['values']['region']
        cap = int(rec['values']['supply_capacity'])
        supply_capacity[region] = cap
    elif rec['source'] == 'transportation_costs.csv':
        region = rec['values']['Unnamed: 0']
        transportation_costs[region] = {}
        for k, v in rec['values'].items():
            if k == 'Unnamed: 0':
                continue
            transportation_costs[region][k] = float(v)
warehouses = list(supply_capacity.keys())
stores = list(customer_demand.keys())
for i in warehouses:
    if i not in transportation_costs:
        raise ValueError(f'Missing transportation costs for warehouse {i}')
    for j in stores:
        if j not in transportation_costs[i]:
            raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
m = gp.Model('GreenMart_Transportation')
x = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((transportation_costs[i][j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == customer_demand[j] for j in stores), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')