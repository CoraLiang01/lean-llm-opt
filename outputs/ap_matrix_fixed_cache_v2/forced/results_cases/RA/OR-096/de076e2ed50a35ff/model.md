Let:
- $S$ = set of schools = $\{\text{I}, \text{II}\}$
- $N$ = set of neighborhoods = $\{\text{N01}, \text{N02}, \ldots, \text{N31}\}$
- $x_{s,n}^W$ = number of white students assigned from neighborhood $n$ to school $s$
- $x_{s,n}^{NW}$ = number of nonwhite students assigned from neighborhood $n$ to school $s$

Parameters (from data):

- School capacities:
  - $\text{I}$: $2028$
  - $\text{II}$: $1560$
- Neighborhood populations (for each $n$):
  - $P_n^W$ = Population_White
  - $P_n^{NW}$ = Population_NonWhite
- Distances $d_{s,n}$ (in miles) from each school $s$ to each neighborhood $n$ (see below)
- District-wide totals:
  - $P^W = \sum_{n \in N} P_n^W$
  - $P^{NW} = \sum_{n \in N} P_n^{NW}$
  - District white percentage: $r = \frac{P^W}{P^W + P^{NW}} = 0.6$
  - Allowable deviation: $\pm 0.10$ (i.e., school white percentage must be in $[0.5, 0.7]$)

#### Objective

Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} d_{s,n} \left( x_{s,n}^W + x_{s,n}^{NW} \right)
$$

#### Constraints

1. **Neighborhood assignment:**
   - All white and nonwhite students from each neighborhood must be assigned:
     $$
     \sum_{s \in S} x_{s,n}^W = P_n^W, \quad \forall n \in N
     $$
     $$
     \sum_{s \in S} x_{s,n}^{NW} = P_n^{NW}, \quad \forall n \in N
     $$

2. **School capacity:**
   - Total students assigned to each school cannot exceed its capacity:
     $$
     \sum_{n \in N} \left( x_{s,n}^W + x_{s,n}^{NW} \right) \leq \text{Capacity}_s, \quad \forall s \in S
     $$
     where $\text{Capacity}_\text{I} = 2028$, $\text{Capacity}_\text{II} = 1560$

3. **Racial balance at each school:**
   - The white percentage at each school must be within $[0.5, 0.7]$:
     $$
     0.5 \leq \frac{\sum_{n \in N} x_{s,n}^W}{\sum_{n \in N} \left( x_{s,n}^W + x_{s,n}^{NW} \right)} \leq 0.7, \quad \forall s \in S
     $$
     For linearization, for each $s$:
     $$
     \sum_{n \in N} x_{s,n}^W \geq 0.5 \sum_{n \in N} \left( x_{s,n}^W + x_{s,n}^{NW} \right)
     $$
     $$
     \sum_{n \in N} x_{s,n}^W \leq 0.7 \sum_{n \in N} \left( x_{s,n}^W + x_{s,n}^{NW} \right)
     $$

4. **Nonnegativity and integrality:**
   $$
   x_{s,n}^W \in \mathbb{Z}_{\geq 0}, \quad x_{s,n}^{NW} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S, n \in N
   $$

---

#### Data (as retrieved):

- **Schools and capacities:**
  - I: 2028
  - II: 1560

- **Neighborhoods and populations:**

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

- **Distances (miles):**

| School | N01  | N02  | N03  | N04  | N05  | N06  | N07  | N08  | N09  | N10  | N11  | N12  | N13  | N14  | N15  | N16  | N17  | N18  | N19  | N20  | N21  | N22  | N23  | N24  | N25  | N26  | N27  | N28  | N29  | N30  | N31  |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| I      | 1.25 | 1.3  | 1.35 | 1.4  | 1.45 | 1.5  | 1.55 | 1.6  | 1.65 | 1.7  | 1.75 | 1.8  | 1.85 | 1.9  | 1.95 | 2.0  | 3.08 | 3.16 | 3.24 | 3.32 | 3.4  | 3.48 | 3.56 | 3.64 | 3.72 | 3.8  | 3.88 | 3.96 | 4.04 | 4.12 | 4.2  |
| II     | 3.08 | 3.16 | 3.24 | 3.32 | 3.4  | 3.48 | 3.56 | 3.64 | 3.72 | 3.8  | 3.88 | 3.96 | 4.04 | 4.12 | 4.2  | 4.28 | 1.25 | 1.3  | 1.35 | 1.4  | 1.45 | 1.5  | 1.55 | 1.6  | 1.65 | 1.7  | 1.75 | 1.8  | 1.85 | 1.9  | 1.95 |

---

**Decision variables:**
- $x_{s,n}^W$, $x_{s,n}^{NW}$: nonnegative integers, for all $s \in S$, $n \in N$

**All parameters and constraints are as above, using the exact identifiers and values from the data.**