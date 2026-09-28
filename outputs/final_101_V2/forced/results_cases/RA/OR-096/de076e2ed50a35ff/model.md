Let:
- $S$ = set of schools = {I, II}
- $N$ = set of neighborhoods = {N01, N02, ..., N31}
- $x_{s,n}^W$ = number of white students assigned from neighborhood $n$ to school $s$
- $x_{s,n}^{NW}$ = number of nonwhite students assigned from neighborhood $n$ to school $s$

Parameters (from data):

School capacities:
- $C_I = 2028$
- $C_{II} = 1560$

Neighborhood populations (for each $n \in N$):
- $P_n^W$ = Population_White in $n$
- $P_n^{NW}$ = Population_NonWhite in $n$

Distances (for each $s \in S$, $n \in N$):
- $d_{s,n}$ = distance from school $s$ to neighborhood $n$

District-wide totals:
- $P^W = \sum_{n \in N} P_n^W = 1567$
- $P^{NW} = \sum_{n \in N} P_n^{NW} = 1100$
- District white percentage: $r = \frac{P^W}{P^W + P^{NW}} = \frac{1567}{2667} \approx 0.5876$ (but use 60% as per the question)
- Allowable deviation: $\pm 10\%$ (i.e., white percentage at each school must be between 50% and 70%)

Objective:
Minimize total travel distance:
$$
\min \sum_{s \in S} \sum_{n \in N} d_{s,n} \left( x_{s,n}^W + x_{s,n}^{NW} \right)
$$

Subject to:

1. Neighborhood assignment constraints (all students assigned):
$$
\sum_{s \in S} x_{s,n}^W = P_n^W, \quad \forall n \in N
$$
$$
\sum_{s \in S} x_{s,n}^{NW} = P_n^{NW}, \quad \forall n \in N
$$

2. School capacity constraints:
$$
\sum_{n \in N} \left( x_{s,n}^W + x_{s,n}^{NW} \right) \leq C_s, \quad \forall s \in S
$$

3. Racial balance constraints (for each school $s$):
Let $T_s = \sum_{n \in N} \left( x_{s,n}^W + x_{s,n}^{NW} \right)$ (total students at $s$)
Let $W_s = \sum_{n \in N} x_{s,n}^W$ (white students at $s$)

Require:
$$
0.5 \leq \frac{W_s}{T_s} \leq 0.7, \quad \forall s \in S, \quad \text{if } T_s > 0
$$
Or, equivalently (for $T_s > 0$):
$$
0.5 T_s \leq W_s \leq 0.7 T_s, \quad \forall s \in S
$$

4. Nonnegativity and integrality:
$$
x_{s,n}^W, \ x_{s,n}^{NW} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S, \ n \in N
$$

---

#### Data (as retrieved):

school_capacity.csv
- I: 2028
- II: 1560

neighborhoods_population.csv (abbreviated for space; full data in source order)
- N01: 78 white, 22 nonwhite
- N02: 57 white, 33 nonwhite
- ...
- N31: 74 white, 46 nonwhite

distance.csv (for each school, all neighborhoods, in source order)
- I: N01=1.25, N02=1.3, ..., N31=4.2
- II: N01=3.08, N02=3.16, ..., N31=1.95

---

#### Complete Mathematical Model

Minimize:
$$
\sum_{s \in \{\text{I}, \text{II}\}} \sum_{n \in \{\text{N01}, \ldots, \text{N31}\}} d_{s,n} \left( x_{s,n}^W + x_{s,n}^{NW} \right)
$$

Subject to:
$$
\sum_{s \in \{\text{I}, \text{II}\}} x_{s,n}^W = P_n^W, \quad \forall n \in \{\text{N01}, \ldots, \text{N31}\}
$$
$$
\sum_{s \in \{\text{I}, \text{II}\}} x_{s,n}^{NW} = P_n^{NW}, \quad \forall n \in \{\text{N01}, \ldots, \text{N31}\}
$$
$$
\sum_{n \in \{\text{N01}, \ldots, \text{N31}\}} \left( x_{s,n}^W + x_{s,n}^{NW} \right) \leq C_s, \quad \forall s \in \{\text{I}, \text{II}\}
$$
$$
0.5 \sum_{n} \left( x_{s,n}^W + x_{s,n}^{NW} \right) \leq \sum_{n} x_{s,n}^W \leq 0.7 \sum_{n} \left( x_{s,n}^W + x_{s,n}^{NW} \right), \quad \forall s \in \{\text{I}, \text{II}\}
$$
$$
x_{s,n}^W, \ x_{s,n}^{NW} \in \mathbb{Z}_{\geq 0}, \quad \forall s, n
$$

All parameters ($C_s$, $P_n^W$, $P_n^{NW}$, $d_{s,n}$) are as given in the retrieved data, in original source order.