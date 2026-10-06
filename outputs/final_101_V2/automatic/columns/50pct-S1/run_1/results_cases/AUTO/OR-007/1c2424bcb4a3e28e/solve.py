LEGACY_OBSERVATION = 'archive_revision_number,ProductName,Value,record_keeper_group,Weight\n9,,765,,\n7,Sedan,2524,Team A,99\n3,SUV,4614,Team C,55\n9,Truck,8416,Team B,75\n8,Convertible,5917,Team C,94\n4,Minivan,9048,Team B,80\n6,Coupe,1140,Team B,82\n5,Hatchback,8962,Team C,71\n7,Station Wagon,1888,Team C,100\n8,Electric Car,8487,Team B,28\n1,Hybrid Car,4425,Team C,93\n8,Luxury Sedan,4717,Team C,84\n1,Sports Car,4210,Team A,83\n8,Crossover,1226,Team B,62\n7,Diesel Truck,7400,Team B,90\n6,Compact SUV,4639,Team B,99\n9,Luxury SUV,7712,Team B,96\n4,Cargo Van,3299,Team C,21\n8,Pickup Truck,9895,Team A,39\n7,Roadster,4496,Team A,99\n3,Muscle Car,4526,Team B,81\n5,Off-road Vehicle,5688,Team C,6\n1,Camper Van,3007,Team C,58\n4,Compact Car,3623,Team A,37\n8,Motorcycle,8474,Team C,15\n3,Electric SUV,8372,Team B,37'
LEGACY_RECORDS = [{'source': '', 'values': {'archive_revision_number': '9', 'ProductName': '', 'Value': '765', 'record_keeper_group': '', 'Weight': ''}}, {'source': '', 'values': {'archive_revision_number': '7', 'ProductName': 'Sedan', 'Value': '2524', 'record_keeper_group': 'Team A', 'Weight': '99'}}, {'source': '', 'values': {'archive_revision_number': '3', 'ProductName': 'SUV', 'Value': '4614', 'record_keeper_group': 'Team C', 'Weight': '55'}}, {'source': '', 'values': {'archive_revision_number': '9', 'ProductName': 'Truck', 'Value': '8416', 'record_keeper_group': 'Team B', 'Weight': '75'}}, {'source': '', 'values': {'archive_revision_number': '8', 'ProductName': 'Convertible', 'Value': '5917', 'record_keeper_group': 'Team C', 'Weight': '94'}}, {'source': '', 'values': {'archive_revision_number': '4', 'ProductName': 'Minivan', 'Value': '9048', 'record_keeper_group': 'Team B', 'Weight': '80'}}, {'source': '', 'values': {'archive_revision_number': '6', 'ProductName': 'Coupe', 'Value': '1140', 'record_keeper_group': 'Team B', 'Weight': '82'}}, {'source': '', 'values': {'archive_revision_number': '5', 'ProductName': 'Hatchback', 'Value': '8962', 'record_keeper_group': 'Team C', 'Weight': '71'}}, {'source': '', 'values': {'archive_revision_number': '7', 'ProductName': 'Station Wagon', 'Value': '1888', 'record_keeper_group': 'Team C', 'Weight': '100'}}, {'source': '', 'values': {'archive_revision_number': '8', 'ProductName': 'Electric Car', 'Value': '8487', 'record_keeper_group': 'Team B', 'Weight': '28'}}, {'source': '', 'values': {'archive_revision_number': '1', 'ProductName': 'Hybrid Car', 'Value': '4425', 'record_keeper_group': 'Team C', 'Weight': '93'}}, {'source': '', 'values': {'archive_revision_number': '8', 'ProductName': 'Luxury Sedan', 'Value': '4717', 'record_keeper_group': 'Team C', 'Weight': '84'}}, {'source': '', 'values': {'archive_revision_number': '1', 'ProductName': 'Sports Car', 'Value': '4210', 'record_keeper_group': 'Team A', 'Weight': '83'}}, {'source': '', 'values': {'archive_revision_number': '8', 'ProductName': 'Crossover', 'Value': '1226', 'record_keeper_group': 'Team B', 'Weight': '62'}}, {'source': '', 'values': {'archive_revision_number': '7', 'ProductName': 'Diesel Truck', 'Value': '7400', 'record_keeper_group': 'Team B', 'Weight': '90'}}, {'source': '', 'values': {'archive_revision_number': '6', 'ProductName': 'Compact SUV', 'Value': '4639', 'record_keeper_group': 'Team B', 'Weight': '99'}}, {'source': '', 'values': {'archive_revision_number': '9', 'ProductName': 'Luxury SUV', 'Value': '7712', 'record_keeper_group': 'Team B', 'Weight': '96'}}, {'source': '', 'values': {'archive_revision_number': '4', 'ProductName': 'Cargo Van', 'Value': '3299', 'record_keeper_group': 'Team C', 'Weight': '21'}}, {'source': '', 'values': {'archive_revision_number': '8', 'ProductName': 'Pickup Truck', 'Value': '9895', 'record_keeper_group': 'Team A', 'Weight': '39'}}, {'source': '', 'values': {'archive_revision_number': '7', 'ProductName': 'Roadster', 'Value': '4496', 'record_keeper_group': 'Team A', 'Weight': '99'}}, {'source': '', 'values': {'archive_revision_number': '3', 'ProductName': 'Muscle Car', 'Value': '4526', 'record_keeper_group': 'Team B', 'Weight': '81'}}, {'source': '', 'values': {'archive_revision_number': '5', 'ProductName': 'Off-road Vehicle', 'Value': '5688', 'record_keeper_group': 'Team C', 'Weight': '6'}}, {'source': '', 'values': {'archive_revision_number': '1', 'ProductName': 'Camper Van', 'Value': '3007', 'record_keeper_group': 'Team C', 'Weight': '58'}}, {'source': '', 'values': {'archive_revision_number': '4', 'ProductName': 'Compact Car', 'Value': '3623', 'record_keeper_group': 'Team A', 'Weight': '37'}}, {'source': '', 'values': {'archive_revision_number': '8', 'ProductName': 'Motorcycle', 'Value': '8474', 'record_keeper_group': 'Team C', 'Weight': '15'}}, {'source': '', 'values': {'archive_revision_number': '3', 'ProductName': 'Electric SUV', 'Value': '8372', 'record_keeper_group': 'Team B', 'Weight': '37'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_types = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    pname = v.get('ProductName', '').strip()
    if pname:
        vehicle_types.append(pname)
        try:
            value[pname] = int(v['Value'])
        except Exception:
            raise ValueError(f'Missing or invalid Value for {pname}')
        try:
            weight[pname] = int(v['Weight'])
        except Exception:
            raise ValueError(f'Missing or invalid Weight for {pname}')
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        vals = rec['values']
        for k in vals:
            try:
                capacity = int(vals[k])
                break
            except Exception:
                continue
        if capacity is not None:
            break
if capacity is None:
    raise ValueError("Overall inventory capacity limit C not found in LEGACY_RECORDS (expected in a record with source 'capacity.csv').")
m = gp.Model('vehicle_inventory')
x = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in vehicle_types)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')