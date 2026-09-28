##### Sets

- $W = \{1,2,\ldots,12\}$: set of workers (indexed by $w$)
- $T = \{\text{A},\text{B},\text{C},\text{D},\text{E},\text{F},\text{G},\text{H},\text{I},\text{J}\}$: set of tasks (indexed by $t$)

##### Parameters

Let $c_{wt}$ be the time required for worker $w$ to complete task $t$, as given below:

| $w$ | $c_{wA}$ | $c_{wB}$ | $c_{wC}$ | $c_{wD}$ | $c_{wE}$ | $c_{wF}$ | $c_{wG}$ | $c_{wH}$ | $c_{wI}$ | $c_{wJ}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 4 | 3 | 7 | 6 | 5 | 6 | 3 | 7 | 5 |
| 2 | 4 | 6 | 5 | 6 | 4 | 5 | 3 | 8 | 7 | 6 |
| 3 | 5 | 4 | 7 | 5 | 6 | 6 | 5 | 8 | 6 | 9 |
| 4 | 7 | 5 | 2 | 3 | 7 | 8 | 5 | 6 | 8 | 5 |
| 5 | 10 | 6 | 7 | 4 | 5 | 4 | 4 | 5 | 9 | 7 |
| 6 | 6 | 7 | 6 | 3 | 9 | 5 | 7 | 4 | 3 | 4 |
| 7 | 8 | 8 | 5 | 9 | 5 | 7 | 5 | 9 | 5 | 3 |
| 8 | 7 | 4 | 8 | 8 | 6 | 7 | 5 | 7 | 7 | 7 |
| 9 | 5 | 6 | 8 | 7 | 7 | 8 | 7 | 8 | 4 | 5 |
| 10 | 8 | 7 | 9 | 5 | 8 | 5 | 9 | 9 | 3 | 4 |
| 11 | 9 | 8 | 10 | 8 | 5 | 4 | 7 | 6 | 8 | 7 |
| 12 | 8 | 5 | 6 | 9 | 4 | 7 | 8 | 4 | 7 | 9 |

##### Decision Variables

- $x_{wt} \in \{0,1\}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise
- $y_w \in \{0,1\}$: 1 if worker $w$ is selected (assigned to any task), 0 otherwise

##### Objective

Minimize total working hours:
$$
\min \sum_{w=1}^{12} \sum_{t \in T} c_{wt} x_{wt}
$$

##### Constraints

1. Each task is assigned to exactly one worker:
   $$
   \sum_{w=1}^{12} x_{wt} = 1 \quad \forall t \in T
   $$
2. Each selected worker is assigned to at most one task:
   $$
   \sum_{t \in T} x_{wt} \leq y_w \quad \forall w = 1,\ldots,12
   $$
3. Exactly 10 workers are selected:
   $$
   \sum_{w=1}^{12} y_w = 10
   $$
4. Each worker is assigned to at most one task:
   $$
   \sum_{t \in T} x_{wt} \leq 1 \quad \forall w = 1,\ldots,12
   $$
5. Binary variables:
   $$
   x_{wt} \in \{0,1\} \quad \forall w = 1,\ldots,12,\ t \in T
   $$
   $$
   y_w \in \{0,1\} \quad \forall w = 1,\ldots,12
   $$

##### Data Table

| Worker $w$ | $c_{wA}$ | $c_{wB}$ | $c_{wC}$ | $c_{wD}$ | $c_{wE}$ | $c_{wF}$ | $c_{wG}$ | $c_{wH}$ | $c_{wI}$ | $c_{wJ}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 4 | 3 | 7 | 6 | 5 | 6 | 3 | 7 | 5 |
| 2 | 4 | 6 | 5 | 6 | 4 | 5 | 3 | 8 | 7 | 6 |
| 3 | 5 | 4 | 7 | 5 | 6 | 6 | 5 | 8 | 6 | 9 |
| 4 | 7 | 5 | 2 | 3 | 7 | 8 | 5 | 6 | 8 | 5 |
| 5 | 10 | 6 | 7 | 4 | 5 | 4 | 4 | 5 | 9 | 7 |
| 6 | 6 | 7 | 6 | 3 | 9 | 5 | 7 | 4 | 3 | 4 |
| 7 | 8 | 8 | 5 | 9 | 5 | 7 | 5 | 9 | 5 | 3 |
| 8 | 7 | 4 | 8 | 8 | 6 | 7 | 5 | 7 | 7 | 7 |
| 9 | 5 | 6 | 8 | 7 | 7 | 8 | 7 | 8 | 4 | 5 |
| 10 | 8 | 7 | 9 | 5 | 8 | 5 | 9 | 9 | 3 | 4 |
| 11 | 9 | 8 | 10 | 8 | 5 | 4 | 7 | 6 | 8 | 7 |
| 12 | 8 | 5 | 6 | 9 | 4 | 7 | 8 | 4 | 7 | 9 |