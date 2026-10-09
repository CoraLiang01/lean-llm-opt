import gurobipy as gp
from gurobipy import GRB
workers = ['W00', 'W01', 'W04', 'W06', 'W10', 'W11', 'W14', 'W15', 'W19', 'W20']
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
eligible_pairs = [('W10', 'P00'), ('W10', 'P01'), ('W10', 'P03'), ('W10', 'P04'), ('W10', 'P07'), ('W19', 'P00'), ('W19', 'P02'), ('W19', 'P03'), ('W19', 'P04'), ('W19', 'P05'), ('W19', 'P06'), ('W19', 'P07'), ('W11', 'P00'), ('W11', 'P02'), ('W11', 'P03'), ('W11', 'P04'), ('W11', 'P05'), ('W11', 'P07'), ('W00', 'P01'), ('W00', 'P04'), ('W00', 'P05'), ('W00', 'P06'), ('W00', 'P07'), ('W20', 'P00'), ('W20', 'P01'), ('W20', 'P02'), ('W20', 'P03'), ('W20', 'P06'), ('W20', 'P07'), ('W04', 'P00'), ('W04', 'P01'), ('W04', 'P02'), ('W04', 'P03'), ('W04', 'P04'), ('W04', 'P05'), ('W04', 'P06'), ('W04', 'P07'), ('W01', 'P00'), ('W01', 'P02'), ('W01', 'P05'), ('W01', 'P06'), ('W01', 'P07'), ('W06', 'P00'), ('W06', 'P01'), ('W06', 'P02'), ('W06', 'P03'), ('W06', 'P04'), ('W06', 'P05'), ('W06', 'P06'), ('W06', 'P07'), ('W14', 'P01'), ('W14', 'P03'), ('W14', 'P04'), ('W14', 'P05'), ('W14', 'P06'), ('W15', 'P00'), ('W15', 'P02'), ('W15', 'P03'), ('W15', 'P04'), ('W15', 'P05'), ('W15', 'P06'), ('W15', 'P07')]
cost_cents = {('W10', 'P00'): 147, ('W10', 'P01'): 114, ('W10', 'P03'): 998, ('W10', 'P04'): 1270, ('W10', 'P07'): 953, ('W19', 'P00'): 5, ('W19', 'P02'): 9, ('W19', 'P03'): 109, ('W19', 'P04'): 1306, ('W19', 'P05'): 626, ('W19', 'P06'): 466, ('W19', 'P07'): 1365, ('W11', 'P00'): 235, ('W11', 'P02'): 1034, ('W11', 'P03'): 782, ('W11', 'P04'): 556, ('W11', 'P05'): 1018, ('W11', 'P07'): 1136, ('W00', 'P01'): 17, ('W00', 'P04'): 430, ('W00', 'P05'): 1385, ('W00', 'P06'): 260, ('W00', 'P07'): 1107, ('W20', 'P00'): 675, ('W20', 'P01'): 1055, ('W20', 'P02'): 8, ('W20', 'P03'): 703, ('W20', 'P06'): 893, ('W20', 'P07'): 129, ('W04', 'P00'): 4, ('W04', 'P01'): 8, ('W04', 'P02'): 16, ('W04', 'P03'): 543, ('W04', 'P04'): 205, ('W04', 'P05'): 1149, ('W04', 'P06'): 533, ('W04', 'P07'): 732, ('W01', 'P00'): 19, ('W01', 'P02'): 6, ('W01', 'P05'): 19, ('W01', 'P06'): 8, ('W01', 'P07'): 13, ('W06', 'P00'): 1217, ('W06', 'P01'): 425, ('W06', 'P02'): 1320, ('W06', 'P03'): 1097, ('W06', 'P04'): 822, ('W06', 'P05'): 221, ('W06', 'P06'): 496, ('W06', 'P07'): 908, ('W14', 'P01'): 1361, ('W14', 'P03'): 379, ('W14', 'P04'): 239, ('W14', 'P05'): 460, ('W14', 'P06'): 130, ('W15', 'P00'): 722, ('W15', 'P02'): 577, ('W15', 'P03'): 1062, ('W15', 'P04'): 1314, ('W15', 'P05'): 528, ('W15', 'P06'): 835, ('W15', 'P07'): 918}
for pair in eligible_pairs:
    if pair not in cost_cents:
        raise ValueError(f'Missing cost for eligible pair {pair}')
m = gp.Model('assignment')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_cents[w_p] * x_vars[w_p] for w_p in eligible_pairs)), GRB.MINIMIZE)
for p in projects:
    eligible_workers = [w for w in workers if (w, p) in eligible_pairs]
    m.addConstr(gp.quicksum((x_vars[w, p] for w in eligible_workers)) == 1)
for w in workers:
    eligible_projects = [p for p in projects if (w, p) in eligible_pairs]
    m.addConstr(gp.quicksum((x_vars[w, p] for p in eligible_projects)) <= 1)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')