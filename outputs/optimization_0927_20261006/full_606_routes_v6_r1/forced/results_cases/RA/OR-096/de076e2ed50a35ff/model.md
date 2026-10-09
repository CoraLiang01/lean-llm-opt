**Sets and Indices:**

- Let $S$ be the set of schools: $S = \{\text{I}, \text{II}\}$
- Let $N$ be the set of neighborhoods: $N = \{\text{N01}, \text{N02}, \ldots, \text{N31}\}$
- Let $g \in \{\text{White}, \text{NonWhite}\}$ denote student groups.

**Parameters:**

- $C_s$: Capacity of school $s$ (from school_capacity.csv)
  - $C_{\text{I}} = 2028$
  - $C_{\text{II}} = 1560$
- $P_{n,\text{White}}$: White population in neighborhood $n$ (from neighborhoods_population.csv)
- $P_{n,\text{NonWhite}}$: Nonwhite population in neighborhood $n$ (from neighborhoods_population.csv)
- $d_{s,n}$: Distance in miles from school $s$ to neighborhood $n$ (from distance.csv)

**Decision Variables:**

- $x_{s,n,\text{White}}$: Number of white students assigned from neighborhood $n$ to school $s$
- $x_{s,n,\text{NonWhite}}$: Number of nonwhite students assigned from neighborhood $n$ to school $s$

All $x_{s,n,g} \geq 0$ and integer.

---

### Objective

Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} d_{s,n} \left( x_{s,n,\text{White}} + x_{s,n,\text{NonWhite}} \right)
$$

---

### Constraints

1. **Neighborhood Population Assignment:**

   For each neighborhood $n$:
   $$
   \sum_{s \in S} x_{s,n,\text{White}} = P_{n,\text{White}} \qquad \forall n \in N
   $$
   $$
   \sum_{s \in S} x_{s,n,\text{NonWhite}} = P_{n,\text{NonWhite}} \qquad \forall n \in N
   $$

2. **School Capacity:**

   For each school $s$:
   $$
   \sum_{n \in N} \left( x_{s,n,\text{White}} + x_{s,n,\text{NonWhite}} \right) \leq C_s \qquad \forall s \in S
   $$

3. **Racial Balance at Each School:**

   Let $T_{\text{White}} = \sum_{n \in N} P_{n,\text{White}}$ (total white students in district)

   Let $T_{\text{NonWhite}} = \sum_{n \in N} P_{n,\text{NonWhite}}$ (total nonwhite students in district)

   District white percentage: $r = \dfrac{T_{\text{White}}}{T_{\text{White}} + T_{\text{NonWhite}}} = 0.6$

   For each school $s$:
   $$
   0.5 \leq \frac{\sum_{n \in N} x_{s,n,\text{White}}}{\sum_{n \in N} \left( x_{s,n,\text{White}} + x_{s,n,\text{NonWhite}} \right)} \leq 0.7 \qquad \forall s \in S
   $$
   (i.e., white percentage at each school must be within $[0.5, 0.7]$)

4. **Nonnegativity and Integrality:**

   $$
   x_{s,n,g} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S, n \in N, g \in \{\text{White}, \text{NonWhite}\}
   $$

---

### Data (as retrieved, in source order):

**school_capacity.csv**
| School | Capacity |
|--------|----------|
| I      | 2028     |
| II     | 1560     |

**neighborhoods_population.csv**
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

**distance.csv**
| School | N01 | N02 | N03 | N04 | N05 | N06 | N07 | N08 | N09 | N10 | N11 | N12 | N13 | N14 | N15 | N16 | N17 | N18 | N19 | N20 | N21 | N22 | N23 | N24 | N25 | N26 | N27 | N28 | N29 | N30 | N31 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| I      | 1.25|1.3 |1.35 |1.4 |1.45 |1.5 |1.55 |1.6 |1.65 |1.7 |1.75 |1.8 |1.85 |1.9 |1.95 |2.0 |3.08 |3.16 |3.24 |3.32 |3.4 |3.48 |3.56 |3.64 |3.72 |3.8 |3.88 |3.96 |4.04 |4.12 |4.2 |
| II     |3.08 |3.16 |3.24 |3.32 |3.4 |3.48 |3.56 |3.64 |3.72 |3.8 |3.88 |3.96 |4.04 |4.12 |4.2 |4.28 |1.25 |1.3 |1.35 |1.4 |1.45 |1.5 |1.55 |1.6 |1.65 |1.7 |1.75 |1.8 |1.85 |1.9 |1.95|

---

**Summary of Model:**

- Assign each neighborhood's white and nonwhite students to schools to minimize total travel distance.
- Each neighborhood's full population must be assigned.
- No school may exceed its capacity.
- Each school's white percentage must be within $[0.5, 0.7]$ (i.e., within 10 percentage points of the district's 60% white ratio).
- All assignments are nonnegative integers.