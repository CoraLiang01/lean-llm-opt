LEGACY_OBSERVATION = 'capacity.csv\nStorageID,Capacity\n1,1083\n2,1840\n3,770\n4,1299\n5,1259\n6,543\n7,1831\n8,855\n9,619\n10,637\n11,935\n12,626\n13,1457\n14,1198\n15,837\n\nproducts.csv\nProductName,Value,Weight\nWindow Unit,4811,114\nPortable Unit,1130,200\nSplit System,1611,106\nDuctless System,3368,256\nCentral AC,2135,268\nHybrid AC,1046,185\nGeothermal AC,4030,299\nSmart AC,3761,131\nEvaporative Cooler,3523,139\nPackage Unit,1701,105'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'StorageID': '1', 'Capacity': '1083'}}, {'source': 'capacity.csv', 'values': {'StorageID': '2', 'Capacity': '1840'}}, {'source': 'capacity.csv', 'values': {'StorageID': '3', 'Capacity': '770'}}, {'source': 'capacity.csv', 'values': {'StorageID': '4', 'Capacity': '1299'}}, {'source': 'capacity.csv', 'values': {'StorageID': '5', 'Capacity': '1259'}}, {'source': 'capacity.csv', 'values': {'StorageID': '6', 'Capacity': '543'}}, {'source': 'capacity.csv', 'values': {'StorageID': '7', 'Capacity': '1831'}}, {'source': 'capacity.csv', 'values': {'StorageID': '8', 'Capacity': '855'}}, {'source': 'capacity.csv', 'values': {'StorageID': '9', 'Capacity': '619'}}, {'source': 'capacity.csv', 'values': {'StorageID': '10', 'Capacity': '637'}}, {'source': 'capacity.csv', 'values': {'StorageID': '11', 'Capacity': '935'}}, {'source': 'capacity.csv', 'values': {'StorageID': '12', 'Capacity': '626'}}, {'source': 'capacity.csv', 'values': {'StorageID': '13', 'Capacity': '1457'}}, {'source': 'capacity.csv', 'values': {'StorageID': '14', 'Capacity': '1198'}}, {'source': 'capacity.csv', 'values': {'StorageID': '15', 'Capacity': '837'}}, {'source': 'products.csv', 'values': {'ProductName': 'Window Unit', 'Value': '4811', 'Weight': '114'}}, {'source': 'products.csv', 'values': {'ProductName': 'Portable Unit', 'Value': '1130', 'Weight': '200'}}, {'source': 'products.csv', 'values': {'ProductName': 'Split System', 'Value': '1611', 'Weight': '106'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ductless System', 'Value': '3368', 'Weight': '256'}}, {'source': 'products.csv', 'values': {'ProductName': 'Central AC', 'Value': '2135', 'Weight': '268'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid AC', 'Value': '1046', 'Weight': '185'}}, {'source': 'products.csv', 'values': {'ProductName': 'Geothermal AC', 'Value': '4030', 'Weight': '299'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smart AC', 'Value': '3761', 'Weight': '131'}}, {'source': 'products.csv', 'values': {'ProductName': 'Evaporative Cooler', 'Value': '3523', 'Weight': '139'}}, {'source': 'products.csv', 'values': {'ProductName': 'Package Unit', 'Value': '1701', 'Weight': '105'}}]
from gurobipy import Model, GRB

def solve_amazon_ac_allocation():
    global LEGACY_RECORDS
    storage_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    storage_ids = []
    capacities = {}
    for rec in storage_records:
        sid = int(rec['values']['StorageID'])
        cap = int(rec['values']['Capacity'])
        storage_ids.append(sid)
        capacities[sid] = cap
    product_names = []
    values = {}
    weights = {}
    for rec in product_records:
        pname = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        wt = int(rec['values']['Weight'])
        product_names.append(pname)
        values[pname] = val
        weights[pname] = wt
    product_index_to_name = {1: 'Window Unit', 2: 'Portable Unit', 3: 'Split System', 4: 'Ductless System', 5: 'Central AC', 6: 'Hybrid AC', 7: 'Geothermal AC', 8: 'Smart AC', 9: 'Evaporative Cooler', 10: 'Package Unit'}
    for idx in range(1, 11):
        if product_index_to_name[idx] not in product_names:
            raise ValueError(f'Missing product: {product_index_to_name[idx]}')
    for sid in range(1, 16):
        if sid not in storage_ids:
            raise ValueError(f'Missing storage area: {sid}')
    I = storage_ids
    J = list(range(1, 11))
    v = {j: values[product_index_to_name[j]] for j in J}
    w = {j: weights[product_index_to_name[j]] for j in J}
    C = {i: capacities[i] for i in I}
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, J, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((v[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(sum((w[j] * x[i, j] for j in J)) <= C[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            for j in J:
                var = x[i, j]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_amazon_ac_allocation()