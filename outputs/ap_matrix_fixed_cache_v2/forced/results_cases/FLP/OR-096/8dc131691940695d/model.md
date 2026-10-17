##### Sets and Indices

- $S = \{\text{I}, \text{II}\}$: set of schools, indexed by $s$
- $N = \{\text{N01}, \text{N02}, \ldots, \text{N31}\}$: set of neighborhoods, indexed by $n$
- $R = \{\text{White}, \text{NonWhite}\}$: set of racial groups, indexed by $r$

##### Parameters

- School capacities:
  - $\text{Capacity}_\text{I} = 2028$
  - $\text{Capacity}_\text{II} = 1560$

- Neighborhood populations (for each $n \in N$):
  - $\text{Population}_n^\text{White}$ and $\text{Population}_n^\text{NonWhite}$

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

- Distances (in miles) from each school to each neighborhood:

| School | N01 | N02 | N03 | N04 | N05 | N06 | N07 | N08 | N09 | N10 | N11 | N12 | N13 | N14 | N15 | N16 | N17 | N18 | N19 | N20 | N21 | N22 | N23 | N24 | N25 | N26 | N27 | N28 | N29 | N30 | N31 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| I      | 1.25|1.3 |1.35 |1.4 |1.45 |1.5 |1.55 |1.6 |1.65 |1.7 |1.75 |1.8 |1.85 |1.9 |1.95 |2.0 |3.08 |3.16 |3.24 |3.32 |3.4 |3.48 |3.56 |3.64 |3.72 |3.8 |3.88 |3.96 |4.04 |4.12 |4.2 |
| II     |3.08 |3.16|3.24 |3.32|3.4  |3.48|3.56 |3.64|3.72 |3.8 |3.88 |3.96|4.04 |4.12|4.2  |4.28|1.25 |1.3  |1.35 |1.4  |1.45|1.5  |1.55 |1.6  |1.65 |1.7 |1.75 |1.8  |1.85 |1.9  |1.95|

##### District-wide racial ratio

- Total white students: $W = \sum_{n \in N} \text{Population}_n^\text{White}$
- Total nonwhite students: $NW = \sum_{n \in N} \text{Population}_n^\text{NonWhite}$
- Total students: $T = W + NW$
- District white ratio: $p = W / T = 0.6$ (given)
- Allowed deviation: $\pm 0.10$ (i.e., each school’s white percentage must be in $[0.5, 0.7]$)

##### Decision Variables

- $x_{s,n}^\text{White} \geq 0$: Number of white students assigned from neighborhood $n$ to school $s$
- $x_{s,n}^\text{NonWhite} \geq 0$: Number of nonwhite students assigned from neighborhood $n$ to school $s$

##### Objective Function

Minimize the total distance traveled by all students:
\[
\min \sum_{s \in S} \sum_{n \in N} \left( d_{s,n} \cdot x_{s,n}^\text{White} + d_{s,n} \cdot x_{s,n}^\text{NonWhite} \right)
\]
where $d_{s,n}$ is the distance from school $s$ to neighborhood $n$ (see table above).

##### Constraints

1. **Neighborhood population assignment:**  
   For each neighborhood $n$:
   \[
   \sum_{s \in S} x_{s,n}^\text{White} = \text{Population}_n^\text{White} \qquad \forall n \in N
   \]
   \[
   \sum_{s \in S} x_{s,n}^\text{NonWhite} = \text{Population}_n^\text{NonWhite} \qquad \forall n \in N
   \]

2. **School capacity:**  
   For each school $s$:
   \[
   \sum_{n \in N} \left( x_{s,n}^\text{White} + x_{s,n}^\text{NonWhite} \right) \leq \text{Capacity}_s \qquad \forall s \in S
   \]

3. **Racial balance at each school:**  
   For each school $s$:
   \[
   0.5 \leq \frac{\sum_{n \in N} x_{s,n}^\text{White}}{\sum_{n \in N} \left( x_{s,n}^\text{White} + x_{s,n}^\text{NonWhite} \right)} \leq 0.7 \qquad \forall s \in S, \quad \text{if denominator} > 0
   \]
   (If a school is assigned no students, the ratio is undefined and the constraint is vacuously satisfied.)

4. **Nonnegativity:**
   \[
   x_{s,n}^\text{White} \geq 0, \quad x_{s,n}^\text{NonWhite} \geq 0 \qquad \forall s \in S, n \in N
   \]

##### Summary of Parameters

- $S = \{\text{I}, \text{II}\}$
- $N = \{\text{N01}, \ldots, \text{N31}\}$
- $\text{Capacity}_\text{I} = 2028$, $\text{Capacity}_\text{II} = 1560$
- $\text{Population}_n^\text{White}$, $\text{Population}_n^\text{NonWhite}$ as above
- $d_{s,n}$ as above

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{s \in S} \sum_{n \in N} d_{s,n} \left( x_{s,n}^\text{White} + x_{s,n}^\text{NonWhite} \right) \\
\text{s.t.} \quad
& \sum_{s \in S} x_{s,n}^\text{White} = \text{Population}_n^\text{White} \qquad \forall n \in N \\
& \sum_{s \in S} x_{s,n}^\text{NonWhite} = \text{Population}_n^\text{NonWhite} \qquad \forall n \in N \\
& \sum_{n \in N} \left( x_{s,n}^\text{White} + x_{s,n}^\text{NonWhite} \right) \leq \text{Capacity}_s \qquad \forall s \in S \\
& 0.5 \leq \frac{\sum_{n \in N} x_{s,n}^\text{White}}{\sum_{n \in N} \left( x_{s,n}^\text{White} + x_{s,n}^\text{NonWhite} \right)} \leq 0.7 \qquad \forall s \in S, \text{ if denominator} > 0 \\
& x_{s,n}^\text{White} \geq 0, \quad x_{s,n}^\text{NonWhite} \geq 0 \qquad \forall s \in S, n \in N
\end{align*}
\]

##### All required parameters and data are included as above.