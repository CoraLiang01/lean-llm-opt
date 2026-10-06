import csv
from collections import defaultdict, deque
import gurobipy as gp
from gurobipy import GRB
TARGET="NORTH"
ASOF="2026-09-20"

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def decode(files, scope_column='tenant', scope_value=TARGET, asof=ASOF):
    chosen = {}
    for path in files:
        for row in read_csv(path):
            if row[scope_column] != scope_value or row['effective_date'] > asof:
                continue
            key = row['table'], row['record_id']
            old = chosen.get(key)
            if old is None or int(row['revision']) > int(old['revision']):
                chosen[key] = row
            elif int(row['revision']) == int(old['revision']):
                assert old == row, 'Conflicting retransmissions'
    result = defaultdict(list)
    meta = {'table', scope_column, 'record_id', 'revision', 'effective_date', 'action'}
    for (table, _), row in chosen.items():
        if row['action'] != 'DELETE':
            result[table].append({k: v for k, v in row.items() if k not in meta})
    return result

def money(rows, fx, key):
    out=defaultdict(int)
    for row in rows:
        num,den=fx[row['currency']]
        value=int(row['amount'])*num
        assert value%den==0
        out[row[key]]+=value//den
    return out

def and_var(m,a,b,name):
    z=m.addVar(vtype=GRB.BINARY,name=name)
    m.addConstr(z<=a);m.addConstr(z<=b);m.addConstr(z>=a+b-1)
    return z

def build_model(t, kind):
    fx={r['currency']:(int(r['usd_cents_numerator']),int(r['denominator'])) for r in t['fx']}
    m=gp.Model();m.Params.OutputFlag=0;m.Params.MIPGap=0;m.Params.TimeLimit=120
    if kind=='AP':
        workers={r['worker_ref']:r for r in t['worker']};projects={r['project_ref']:r for r in t['project']}
        routes={r['edge_id']:r for r in t['route']};cost=money(t['charge'],fx,'edge_id')
        assert set(cost)==set(routes)
        rank={s:i for i,s in enumerate(['Junior','Intermediate','Senior','Expert'])}
        x=m.addVars(list(routes),vtype=GRB.BINARY,name='assign')
        for e,r in routes.items():
            w,p=workers[r['worker_ref']],projects[r['project_ref']]
            if int(w['on_leave']) or rank[w['skill']]<rank[p['required_skill']]:m.addConstr(x[e]==0)
        for p in projects:m.addConstr(gp.quicksum(x[e] for e,r in routes.items() if r['project_ref']==p)==1)
        for w in workers:m.addConstr(gp.quicksum(x[e] for e,r in routes.items() if r['worker_ref']==w)<=1)
        objective=gp.quicksum(cost[e]*x[e] for e in routes)
        for team in t['team']:
            ids=[e for e,r in routes.items() if workers[r['worker_ref']]['team']==team['team']]
            count=gp.quicksum(x[e] for e in ids);u=m.addVar(vtype=GRB.BINARY,name='team_'+team['team'])
            m.addConstr(count>=int(team['minimum']));m.addConstr(count<=int(team['maximum']))
            m.addConstr(count<=len(projects)*u);m.addConstr(u<=count)
            m.addConstr(gp.quicksum(int(projects[routes[e]['project_ref']]['effort'])*x[e] for e in ids)<=int(team['workload_limit']))
            objective+=int(team['activation_fee_cents'])*u
        for r in t['conflict']:m.addConstr(x[r['edge_a']]+x[r['edge_b']]<=1)
        for k,r in enumerate(t['pair_adjustment']):objective+=int(r['adjustment_cents'])*and_var(m,x[r['edge_a']],x[r['edge_b']],f'pair{k}')
        m.setObjective(objective,GRB.MINIMIZE)
    else:
        items={r['item_ref']:r for r in t['item']};value=money(t['benefit'],fx,'item_ref')
        assert set(value)==set(items)
        x=m.addVars(list(items),vtype=GRB.INTEGER,lb=0,name='quantity');y=m.addVars(list(items),vtype=GRB.BINARY,name='ordered')
        for i,r in items.items():
            m.addConstr(x[i]>=int(r['minimum_lot'])*y[i]);m.addConstr(x[i]<=int(r['maximum_order'])*y[i])
            if not int(r['authorized']):m.addConstr(y[i]==0)
        objective=gp.quicksum(value[i]*x[i] for i in items)
        for r in t['item_fee']:objective-=int(r['activation_fee_cents'])*y[r['item_ref']]
        scale={'liter':1000,'ml':1,'hour':60,'minute':1,'kwh':1000,'wh':1,'GB':1000,'MB':1}
        caps=defaultdict(int)
        for r in t['capacity_ledger']:caps[r['resource']]+=int(r['amount'])*scale[r['unit']]
        for resource,cap in caps.items():
            m.addConstr(gp.quicksum(int(r['amount'])*scale[r['unit']]*x[r['item_ref']] for r in t['usage'] if r['resource']==resource)<=cap)
        for r in t['category']:
            ids=[i for i,item in items.items() if item['category']==r['category']];count=gp.quicksum(x[i] for i in ids)
            m.addConstr(count>=int(r['minimum_quantity']));m.addConstr(count<=int(r['maximum_quantity']))
            u=m.addVar(vtype=GRB.BINARY,name='category_'+r['category'])
            m.addConstr(gp.quicksum(y[i] for i in ids)<=len(ids)*u);m.addConstr(u<=gp.quicksum(y[i] for i in ids))
            objective-=int(r['activation_fee_cents'])*u
        for r in t['incompatible']:m.addConstr(y[r['item_a']]+y[r['item_b']]<=1)
        for r in t['requires']:m.addConstr(y[r['item_ref']]<=y[r['prerequisite_ref']])
        for k,r in enumerate(t['bundle']):objective+=int(r['bonus_cents'])*and_var(m,y[r['item_a']],y[r['item_b']],f'bundle{k}')
        m.setObjective(objective,GRB.MAXIMIZE)
    m.optimize()
    assert m.Status==GRB.OPTIMAL, (m.Status,m.SolCount)
    solution={i:round(v.X) for i,v in x.items()}
    return m,solution

def audit(t,kind,s):
    """Recompute all business constraints and objective without using model expressions."""
    fx={r['currency']:(int(r['usd_cents_numerator']),int(r['denominator'])) for r in t['fx']}
    if kind=='AP':
        workers={r['worker_ref']:r for r in t['worker']};projects={r['project_ref']:r for r in t['project']}
        routes={r['edge_id']:r for r in t['route']};cost=money(t['charge'],fx,'edge_id');selected={e for e,n in s.items() if n}
        rank={v:k for k,v in enumerate(['Junior','Intermediate','Senior','Expert'])}
        assert all(n in [0,1] for n in s.values())
        counts=defaultdict(int);used=defaultdict(int);tc=defaultdict(int);effort=defaultdict(int)
        for e in selected:
            r=routes[e];w=workers[r['worker_ref']];p=projects[r['project_ref']]
            assert not int(w['on_leave']) and rank[w['skill']]>=rank[p['required_skill']]
            counts[r['project_ref']]+=1;used[r['worker_ref']]+=1;tc[w['team']]+=1;effort[w['team']]+=int(p['effort'])
        assert all(counts[p]==1 for p in projects) and all(n<=1 for n in used.values())
        obj=sum(cost[e] for e in selected)
        for r in t['team']:
            c=tc[r['team']];assert int(r['minimum'])<=c<=int(r['maximum']) and effort[r['team']]<=int(r['workload_limit'])
            if c:obj+=int(r['activation_fee_cents'])
        for r in t['conflict']:assert not {r['edge_a'],r['edge_b']}<=selected
        for r in t['pair_adjustment']:
            if {r['edge_a'],r['edge_b']}<=selected:obj+=int(r['adjustment_cents'])
    else:
        items={r['item_ref']:r for r in t['item']};values=money(t['benefit'],fx,'item_ref');selected={i for i,n in s.items() if n}
        for i,n in s.items():
            assert n>=0
            if n:assert int(items[i]['authorized']) and int(items[i]['minimum_lot'])<=n<=int(items[i]['maximum_order'])
        scale={'liter':1000,'ml':1,'hour':60,'minute':1,'kwh':1000,'wh':1,'GB':1000,'MB':1};caps=defaultdict(int);use=defaultdict(int)
        for r in t['capacity_ledger']:caps[r['resource']]+=int(r['amount'])*scale[r['unit']]
        for r in t['usage']:use[r['resource']]+=s[r['item_ref']]*int(r['amount'])*scale[r['unit']]
        assert all(use[r]<=cap for r,cap in caps.items())
        obj=sum(s[i]*values[i] for i in items)
        for r in t['item_fee']:
            if r['item_ref'] in selected:obj-=int(r['activation_fee_cents'])
        for r in t['category']:
            count=sum(s[i] for i,item in items.items() if item['category']==r['category'])
            assert int(r['minimum_quantity'])<=count<=int(r['maximum_quantity'])
            if count:obj-=int(r['activation_fee_cents'])
        for r in t['incompatible']:assert not {r['item_a'],r['item_b']}<=selected
        for r in t['requires']:assert r['item_ref'] not in selected or r['prerequisite_ref'] in selected
        for r in t['bundle']:
            if {r['item_a'],r['item_b']}<=selected:obj+=int(r['bonus_cents'])
    return obj

def eligible_costs(t):
    workers={r['worker_ref']:r for r in t['worker']};projects={r['project_ref']:r for r in t['project']}
    rank={s:i for i,s in enumerate(['Junior','Intermediate','Senior','Expert'])}
    fx={r['currency']:(int(r['usd_cents_numerator']),int(r['denominator'])) for r in t['fx']}
    costs=money(t['charge'],fx,'edge_id')
    edges={r['edge_id']:r for r in t['route'] if int(workers[r['worker_ref']]['on_leave'])==0 and rank[workers[r['worker_ref']]['skill']]>=rank[projects[r['project_ref']]['required_skill']]}
    return workers,projects,edges,costs

def independent_assignment(workers,projects,edges,costs):
    """Independent integer min-cost-flow implementation, no solver dependency."""
    graph=defaultdict(list)
    def add(a,b,cost):
        graph[a].append([b,1,cost,len(graph[b])]);graph[b].append([a,0,-cost,len(graph[a])-1])
    for w in workers:add('source',w,0)
    for p in projects:add(p,'sink',0)
    for e,r in edges.items():add(r['worker_ref'],r['project_ref'],costs[e])
    total=0
    for _ in projects:
        dist={'source':0};prev={};queue=deque(['source']);inq={'source'}
        while queue:
            a=queue.popleft();inq.remove(a)
            for i,(b,cap,cost,_) in enumerate(graph[a]):
                if cap and dist[a]+cost<dist.get(b,float('inf')):
                    dist[b]=dist[a]+cost;prev[b]=(a,i)
                    if b not in inq:queue.append(b);inq.add(b)
        assert 'sink' in dist,'Infeasible assignment'
        total+=dist['sink'];b='sink'
        while b!='source':
            a,i=prev[b];edge=graph[a][i];edge[1]-=1;graph[b][edge[3]][1]+=1;b=a
    return total

def solve_ap(t):
    workers,projects,edges,costs=eligible_costs(t)
    m=gp.Model();m.Params.OutputFlag=0;m.Params.MIPGap=0
    x=m.addVars(list(edges),vtype=GRB.BINARY,name='assign')
    for p in projects:m.addConstr(gp.quicksum(x[e] for e,r in edges.items() if r['project_ref']==p)==1)
    for w in workers:m.addConstr(gp.quicksum(x[e] for e,r in edges.items() if r['worker_ref']==w)<=1)
    m.setObjective(gp.quicksum(costs[e]*x[e] for e in edges));m.optimize()
    assert m.Status==GRB.OPTIMAL
    selected={e for e,var in x.items() if var.X>.5}
    assert all(sum(edges[e]['project_ref']==p for e in selected)==1 for p in projects)
    assert all(sum(edges[e]['worker_ref']==w for e in selected)<=1 for w in workers)
    objective=sum(costs[e] for e in selected)
    assert objective==round(m.ObjVal)==independent_assignment(workers,projects,edges,costs)
    return m,{e:int(e in selected) for e in edges},objective
