##### Sets and Indices

- $S = \{\text{I}, \text{II}\}$: set of schools, indexed by $s$
- $N = \{\text{N01}, \ldots, \text{N31}\}$: set of neighborhoods, indexed by $n$

##### Parameters

- $C_s$: capacity of school $s$
  - $C_{\text{I}} = 2028$
  - $C_{\text{II}} = 1560$
- $W_n$: white population in neighborhood $n$ (see table below)
- $B_n$: nonwhite population in neighborhood $n$ (see table below)
- $d_{s,n}$: distance in miles from school $s$ to neighborhood $n$ (see table below)
- $T_W = \sum_{n\in N} W_n = 1642$
- $T_B = \sum_{n\in N} B_n = 1056$
- District white ratio: $r = \frac{T_W}{T_W + T_B} = \frac{1642}{2698} \approx 0.6086$
- Racial balance: each school’s white percentage must be in $[0.5, 0.7]$

##### Decision Variables

- $x_{s,n}^W \geq 0$: number of white students from neighborhood $n$ assigned to school $s$
- $x_{s,n}^B \geq 0$: number of nonwhite students from neighborhood $n$ assigned to school $s$

##### Objective

Minimize total travel distance:
$$
\min \sum_{s\in S} \sum_{n\in N} d_{s,n} \left( x_{s,n}^W + x_{s,n}^B \right)
$$

##### Constraints

1. **Neighborhood assignment:** All students from each neighborhood must be assigned:
   $$
   \sum_{s\in S} x_{s,n}^W = W_n \quad \forall n\in N
   $$
   $$
   \sum_{s\in S} x_{s,n}^B = B_n \quad \forall n\in N
   $$

2. **School capacity:** Each school cannot exceed its capacity:
   $$
   \sum_{n\in N} (x_{s,n}^W + x_{s,n}^B) \leq C_s \quad \forall s\in S
   $$

3. **Racial balance:** For each school, the white percentage must be between 50% and 70%:
   $$
   0.5 \leq \frac{\sum_{n\in N} x_{s,n}^W}{\sum_{n\in N} (x_{s,n}^W + x_{s,n}^B)} \leq 0.7 \quad \forall s\in S
   $$
   (If denominator is zero, the constraint is vacuously satisfied.)

4. **Nonnegativity:**
   $$
   x_{s,n}^W \geq 0,\quad x_{s,n}^B \geq 0 \quad \forall s\in S, n\in N
   $$

##### Data Tables

**School Capacities**

| School | Capacity |
|--------|----------|
| I      | 2028     |
| II     | 1560     |

**Neighborhood Populations**

| Neighborhood | Population_White | Population_NonWhite |
|--------------|------------------|---------------------|
| N01          | 78               | 22                  |
| N02          | 57               | 33                  |
| N03          | 47               | 63                  |
| N04          | 78               | 22                  |
| N05          | 57               | 33                  |
| N06          | 47               | 63                  |
| N07          | 78               | 22                  |
| N08          | 57               | 33                  |
| N09          | 46               | 64                  |
| N10          | 77               | 23                  |
| N11          | 56               | 34                  |
| N12          | 46               | 64                  |
| N13          | 77               | 23                  |
| N14          | 56               | 34                  |
| N15          | 46               | 64                  |
| N16          | 77               | 23                  |
| N17          | 56               | 34                  |
| N18          | 46               | 64                  |
| N19          | 77               | 23                  |
| N20          | 56               | 34                  |
| N21          | 46               | 64                  |
| N22          | 77               | 23                  |
| N23          | 56               | 34                  |
| N24          | 46               | 64                  |
| N25          | 77               | 23                  |
| N26          | 56               | 34                  |
| N27          | 46               | 64                  |
| N28          | 77               | 23                  |
| N29          | 56               | 34                  |
| N30          | 46               | 64                  |
| N31          | 74               | 46                  |

**Distances (in miles)**

| School | N01 | N02 | N03 | N04 | N05 | N06 | N07 | N08 | N09 | N10 | N11 | N12 | N13 | N14 | N15 | N16 | N17 | N18 | N19 | N20 | N21 | N22 | N23 | N24 | N25 | N26 | N27 | N28 | N29 | N30 | N31 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| I      | 1.25|1.3 |1.35 |1.4  |1.45 |1.5  |1.55 |1.6  |1.65 |1.7  |1.75 |1.8  |1.85 |1.9  |1.95 |2.0  |3.08 |3.16 |3.24 |3.32 |3.4  |3.48 |3.56 |3.64 |3.72 |3.8  |3.88 |3.96 |4.04 |4.12 |4.2  |
| II     |3.08 |3.16|3.24 |3.32 |3.4  |3.48 |3.56 |3.64 |3.72 |3.8  |3.88 |3.96 |4.04 |4.12 |4.2  |4.28 |1.25 |1.3  |1.35 |1.4  |1.45 |1.5  |1.55 |1.6  |1.65 |1.7  |1.75 |1.8  |1.85 |1.9  |1.95 |

All data and constraints are included as required.