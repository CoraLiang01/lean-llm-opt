import gurobipy as gp
from gurobipy import GRB
suppliers = ['supply1', 'supply2', 'supply3', 'supply4', 'supply5', 'supply6', 'supply7', 'supply8']
customers = ['demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8']
demand = {'demand1': 9, 'demand2': 66, 'demand3': 56, 'demand4': 17, 'demand5': 43, 'demand6': 62, 'demand7': 10, 'demand8': 37}
supply_capacity = {'supply1': 60, 'supply2': 22, 'supply3': 16, 'supply4': 14, 'supply5': 19, 'supply6': 70, 'supply7': 60, 'supply8': 39}
cost = {'supply1': {'demand1': 0.03020736643461065, 'demand2': 229.50723504640203, 'demand3': 198.62356558205792, 'demand4': 12.995050640153751, 'demand5': 211.20732124396406, 'demand6': 134.9442985029274, 'demand7': 9.822206398831067, 'demand8': 11.394077543225675}, 'supply2': {'demand1': 232.34691308087835, 'demand2': 3.6258726438627473, 'demand3': 0.28605434149404785, 'demand4': 45.73127693242935, 'demand5': 2.8304796563034573, 'demand6': 107.05891033185472, 'demand7': 299.96317913389305, 'demand8': 23.79935436307657}, 'supply3': {'demand1': 11.061938334356302, 'demand2': 0.2041995326579051, 'demand3': 0.2789447278030927, 'demand4': 45.721912724349636, 'demand5': 59.54895565737313, 'demand6': 5.097536739581239, 'demand7': 300.00118415135785, 'demand8': 23.711282707746893}, 'supply4': {'demand1': 235.1794835706472, 'demand2': 43.794668963036194, 'demand3': 40.709846782945924, 'demand4': 0.07774496620087613, 'demand5': 4.237728183419554, 'demand6': 131.70915517494691, 'demand7': 296.55587567706743, 'demand8': 29.810940017561297}, 'supply5': {'demand1': 211.85808746383796, 'demand2': 47.60180876530328, 'demand3': 50.04007716193931, 'demand4': 86.14548807358399, 'demand5': 0.06197897916874956, 'demand6': 5.3345515296262205, 'demand7': 270.06290423798396, 'demand8': 3.853933133973331}, 'supply6': {'demand1': 6.45506633554524, 'demand2': 88.16323623354015, 'demand3': 5.047091671641611, 'demand4': 151.46120287365497, 'demand5': 5.290760161059401, 'demand6': 0.04602205335871525, 'demand7': 9.93670660180487, 'demand8': 103.75460989446313}, 'supply7': {'demand1': 174.27229047340035, 'demand2': 250.58223528739327, 'demand3': 253.90413041857263, 'demand4': 16.235467318386764, 'demand5': 12.643140514778086, 'demand6': 175.0672824108511, 'demand7': 2.983839625303656, 'demand8': 317.0655193866389}, 'supply8': {'demand1': 207.87006253790491, 'demand2': 1.517168471518212, 'demand3': 24.027239288137153, 'demand4': 27.133999276450346, 'demand5': 73.20672468851855, 'demand6': 125.72910359893308, 'demand7': 15.463103251642147, 'demand8': 0.20164987511903337}}
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing cost data for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for supplier {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand data for customer {j}')
for i in suppliers:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity data for supplier {i}')
m = gp.Model('Amazon_Distribution')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')