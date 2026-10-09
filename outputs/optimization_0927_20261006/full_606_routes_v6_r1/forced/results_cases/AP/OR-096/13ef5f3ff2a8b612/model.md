##### Sets and Indices

- Let $S = \{\text{I}, \text{II}\}$ be the set of schools, indexed by $s$.
- Let $N = \{\text{N01}, \text{N02}, \ldots, \text{N31}\}$ be the set of neighborhoods, indexed by $n$.
- Let $R = \{\text{White}, \text{NonWhite}\}$ be the set of racial groups, indexed by $r$.

##### Parameters

- $C_s$: Capacity of school $s$.

  $C_{\text{I}} = 2028$  
  $C_{\text{II}} = 1560$

- $P_{n,r}$: Population of students of race $r$ in neighborhood $n$.

  (See full table below.)

- $d_{s,n}$: Distance in miles from school $s$ to neighborhood $n$.

  (See full table below.)

- District-wide total white students: $W = \sum_{n \in N} P_{n,\text{White}}$
- District-wide total nonwhite students: $NW = \sum_{n \in N} P_{n,\text{NonWhite}}$
- District-wide white percentage: $p_W = \frac{W}{W + NW} = 0.6$
- District-wide nonwhite percentage: $p_{NW} = 0.4$

##### Decision Variables

- $x_{s,n,r} \geq 0$: Number of students of race $r$ from neighborhood $n$ assigned to school $s$.

##### Objective Function

Minimize the total distance traveled by all students:

$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in R} d_{s,n} \cdot x_{s,n,r}
$$

##### Constraints

1. **Neighborhood Population Assignment**

   All students from each neighborhood and race must be assigned to some school:

   $$
   \sum_{s \in S} x_{s,n,r} = P_{n,r} \quad \forall n \in N, \forall r \in R
   $$

2. **School Capacity**

   The total number of students assigned to each school cannot exceed its capacity:

   $$
   \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq C_s \quad \forall s \in S
   $$

3. **Racial Balance at Each School**

   The percentage of white students at each school must be within 10 percentage points of the district-wide percentage (i.e., between 50% and 70%):

   $$
   0.5 \leq \frac{\sum_{n \in N} x_{s,n,\text{White}}}{\sum_{n \in N} \sum_{r \in R} x_{s,n,r}} \leq 0.7 \quad \forall s \in S
   $$

   (If a school receives no students, the denominator is zero; in practice, the model will assign students to both schools.)

4. **Nonnegativity**

   $$
   x_{s,n,r} \geq 0 \quad \forall s \in S, n \in N, r \in R
   $$

---

##### Retrieved Information

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

**Distances (in miles) from each school to each neighborhood**

| School | N01 | N02 | N03 | N04 | N05 | N06 | N07 | N08 | N09 | N10 | N11 | N12 | N13 | N14 | N15 | N16 | N17 | N18 | N19 | N20 | N21 | N22 | N23 | N24 | N25 | N26 | N27 | N28 | N29 | N30 | N31 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| I      | 1.25|1.3 |1.35 |1.4  |1.45 |1.5  |1.55 |1.6  |1.65 |1.7  |1.75 |1.8  |1.85 |1.9  |1.95 |2.0  |3.08 |3.16 |3.24 |3.32 |3.4  |3.48 |3.56 |3.64 |3.72 |3.8  |3.88 |3.96 |4.04 |4.12 |4.2  |
| II     |3.08 |3.16|3.24 |3.32 |3.4  |3.48 |3.56 |3.64 |3.72 |3.8  |3.88 |3.96 |4.04 |4.12 |4.2  |4.28 |1.25 |1.3  |1.35 |1.4  |1.45 |1.5  |1.55 |1.6  |1.65 |1.7  |1.75 |1.8  |1.85 |1.9  |1.95 |

**Neighborhoods:**  
N01, N02, N03, N04, N05, N06, N07, N08, N09, N10, N11, N12, N13, N14, N15, N16, N17, N18, N19, N20, N21, N22, N23, N24, N25, N26, N27, N28, N29, N30, N31

**Schools:**  
I, II

**Racial Groups:**  
White, NonWhite

**Population Table $P_{n,r}$:**  
(see above)

**Distance Table $d_{s,n}$:**  
(see above)

---

This model assigns all white and nonwhite students from each neighborhood to schools, respects school capacities, enforces racial balance at each school, and minimizes total student travel distance.