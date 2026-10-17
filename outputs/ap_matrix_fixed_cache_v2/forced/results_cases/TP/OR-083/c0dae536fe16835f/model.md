Let $W = \{1,2,\ldots,12\}$ be the set of workers and $T = \{A,B,C,D,E,F,G,H,I,J\}$ the set of tasks.

Define binary variables:
- $y_w \in \{0,1\}$: $1$ if worker $w$ is selected (assigned to a task), $0$ otherwise, for $w \in W$.
- $x_{w,t} \in \{0,1\}$: $1$ if worker $w$ is assigned to task $t$, $0$ otherwise, for $w \in W$, $t \in T$.

Parameters: $c_{w,t}$ is the time required for worker $w$ to complete task $t$, as given below.

Minimize total working hours:
$$
\min \sum_{w=1}^{12} \sum_{t \in T} c_{w,t} x_{w,t}
$$

Subject to:
1. Each task is assigned to exactly one worker:
$$
\sum_{w=1}^{12} x_{w,t} = 1 \quad \forall t \in T
$$

2. Each worker is assigned to at most one task:
$$
\sum_{t \in T} x_{w,t} \leq y_w \quad \forall w = 1,\ldots,12
$$

3. Exactly 10 workers are selected:
$$
\sum_{w=1}^{12} y_w = 10
$$

4. Assignment only if selected:
$$
x_{w,t} \leq y_w \quad \forall w = 1,\ldots,12,\ \forall t \in T
$$

5. Binary variables:
$$
x_{w,t} \in \{0,1\} \quad \forall w = 1,\ldots,12,\ \forall t \in T
$$
$$
y_w \in \{0,1\} \quad \forall w = 1,\ldots,12
$$

Parameters $c_{w,t}$ (time required for worker $w$ to complete task $t$):

| Worker |   A |  B |  C |  D |  E |  F |  G |  H |  I |  J |
|--------|----|----|----|----|----|----|----|----|----|----|
|   1    |  9 |  4 |  3 |  7 |  6 |  5 |  6 |  3 |  7 |  5 |
|   2    |  4 |  6 |  5 |  6 |  4 |  5 |  3 |  8 |  7 |  6 |
|   3    |  5 |  4 |  7 |  5 |  6 |  6 |  5 |  8 |  6 |  9 |
|   4    |  7 |  5 |  2 |  3 |  7 |  8 |  5 |  6 |  8 |  5 |
|   5    | 10 |  6 |  7 |  4 |  5 |  4 |  4 |  5 |  9 |  7 |
|   6    |  6 |  7 |  6 |  3 |  9 |  5 |  7 |  4 |  3 |  4 |
|   7    |  8 |  8 |  5 |  9 |  5 |  7 |  5 |  9 |  5 |  3 |
|   8    |  7 |  4 |  8 |  8 |  6 |  7 |  5 |  7 |  7 |  7 |
|   9    |  5 |  6 |  8 |  7 |  7 |  8 |  7 |  8 |  4 |  5 |
|  10    |  8 |  7 |  9 |  5 |  8 |  5 |  9 |  9 |  3 |  4 |
|  11    |  9 |  8 | 10 |  8 |  5 |  4 |  7 |  6 |  8 |  7 |
|  12    |  8 |  5 |  6 |  9 |  4 |  7 |  8 |  4 |  7 |  9 |