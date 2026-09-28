import gurobipy as gp
from gurobipy import GRB
suppliers = ['supply1', 'supply2', 'supply3', 'supply4', 'supply5', 'supply6', 'supply7', 'supply8']
customers = ['demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8']
customer_demand = {'demand1': 9, 'demand2': 66, 'demand3': 56, 'demand4': 17, 'demand5': 43, 'demand6': 62, 'demand7': 10, 'demand8': 37}
supply_capacity = {'supply1': 60, 'supply2': 22, 'supply3': 16, 'supply4': 14, 'supply5': 19, 'supply6': 70, 'supply7': 60, 'supply8': 39}
transportation_costs = {'supply1': {'demand1': 0.0302, 'demand2': 229.5072, 'demand3': 198.6236, 'demand4': 12.9951, 'demand5': 211.2073, 'demand6': 134.9443, 'demand7': 9.8222, 'demand8': 11.3941}, 'supply2': {'demand1': 232.3469, 'demand2': 3.6259, 'demand3': 0.2861, 'demand4': 45.7313, 'demand5': 2.8305, 'demand6': 107.0589, 'demand7': 299.9632, 'demand8': 23.7994}, 'supply3': {'demand1': 11.0619, 'demand2': 0.2042, 'demand3': 0.2789, 'demand4': 45.7219, 'demand5': 59.549, 'demand6': 5.0975, 'demand7': 300.0012, 'demand8': 23.7113}, 'supply4': {'demand1': 235.1795, 'demand2': 43.7947, 'demand3': 40.7098, 'demand4': 0.0777, 'demand5': 4.2377, 'demand6': 131.7092, 'demand7': 296.5559, 'demand8': 29.8109}, 'supply5': {'demand1': 211.8581, 'demand2': 47.6018, 'demand3': 50.0401, 'demand4': 86.1455, 'demand5': 0.062, 'demand6': 5.3346, 'demand7': 270.0629, 'demand8': 3.8539}, 'supply6': {'demand1': 6.4551, 'demand2': 88.1632, 'demand3': 5.0471, 'demand4': 151.4612, 'demand5': 5.2908, 'demand6': 0.046, 'demand7': 9.9367, 'demand8': 103.7546}, 'supply7': {'demand1': 174.2723, 'demand2': 250.5822, 'demand3': 253.9041, 'demand4': 16.2355, 'demand5': 12.6431, 'demand6': 175.0673, 'demand7': 2.9838, 'demand8': 317.0655}, 'supply8': {'demand1': 207.8701, 'demand2': 1.5172, 'demand3': 24.0272, 'demand4': 27.134, 'demand5': 73.2067, 'demand6': 125.7291, 'demand7': 15.4631, 'demand8': 0.2016}}
for i in suppliers:
    if i not in transportation_costs:
        raise ValueError(f'Missing transportation_costs for supplier {i}')
    for j in customers:
        if j not in transportation_costs[i]:
            raise ValueError(f'Missing transportation_costs for supplier {i}, customer {j}')
for j in customers:
    if j not in customer_demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in suppliers:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for supplier {i}')
m = gp.Model('Amazon_Distribution')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((transportation_costs[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == customer_demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')