##### Sets and Indices

- $S = \{\text{I}, \text{II}\}$: set of schools, indexed by $s$
- $N = \{\text{N01}, \ldots, \text{N31}\}$: set of neighborhoods, indexed by $n$
- $R = \{\text{White}, \text{NonWhite}\}$: set of racial groups, indexed by $r$

##### Parameters

- $\text{Capacity}_s$: capacity of school $s$
  - $\text{Capacity}_\text{I} = 2028$
  - $\text{Capacity}_\text{II} = 1560$
- $\text{Pop}_{n,r}$: number of students of race $r$ in neighborhood $n$ (see table below)
- $\text{Dist}_{s,n}$: distance in miles from school $s$ to neighborhood $n$ (see table below)
- District-wide white percentage: $p_\text{white} = 0.6$, $p_\text{nonwhite} = 0.4$
- Allowed deviation: $0.1$ (10 percentage points)

##### Decision Variables

- $x_{s,n,r} \geq 0$: number of students of race $r$ from neighborhood $n$ assigned to school $s$ (continuous or integer, as appropriate)

##### Objective

Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in R} \text{Dist}_{s,n} \cdot x_{s,n,r}
$$

##### Constraints

1. **Neighborhood population assignment:**  
   For all $n \in N$, $r \in R$:
   $$
   \sum_{s \in S} x_{s,n,r} = \text{Pop}_{n,r}
   $$

2. **School capacity:**  
   For all $s \in S$:
   $$
   \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq \text{Capacity}_s
   $$

3. **Racial balance at each school:**  
   For all $s \in S$:
   $$
   0.5 \leq \frac{\sum_{n \in N} x_{s,n,\text{White}}}{\sum_{n \in N} \sum_{r \in R} x_{s,n,r}} \leq 0.7
   $$
   (i.e., white percentage within $[0.6-0.1, 0.6+0.1]$)

4. **Nonnegativity:**  
   $$
   x_{s,n,r} \geq 0 \quad \forall s \in S, n \in N, r \in R
   $$

---

##### Data

**School capacities:**

| School | Capacity |
|--------|----------|
| I      | 2028     |
| II     | 1560     |

**Neighborhood populations:**

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

**Distances (in miles):**

| School | N01 | N02 | N03 | N04 | N05 | N06 | N07 | N08 | N09 | N10 | N11 | N12 | N13 | N14 | N15 | N16 | N17 | N18 | N19 | N20 | N21 | N22 | N23 | N24 | N25 | N26 | N27 | N28 | N29 | N30 | N31 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| I      | 1.25|1.3 |1.35 |1.4  |1.45 |1.5  |1.55 |1.6  |1.65 |1.7  |1.75 |1.8  |1.85 |1.9  |1.95 |2.0  |3.08 |3.16 |3.24 |3.32 |3.4  |3.48 |3.56 |3.64 |3.72 |3.8  |3.88 |3.96 |4.04 |4.12 |4.2  |
| II     |3.08 |3.16|3.24 |3.32 |3.4  |3.48 |3.56 |3.64 |3.72 |3.8  |3.88 |3.96 |4.04 |4.12 |4.2  |4.28 |1.25 |1.3  |1.35 |1.4  |1.45 |1.5  |1.55 |1.6  |1.65 |1.7  |1.75 |1.8  |1.85 |1.9  |1.95 |

---

All identifiers and coefficients are preserved as in the source data.