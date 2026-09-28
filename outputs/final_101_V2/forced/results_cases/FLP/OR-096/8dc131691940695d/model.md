##### Decision Variables

Let:
- $x_{sk}^w \geq 0$: Number of white students assigned from neighborhood $k \in K$ to school $s \in S$.
- $x_{sk}^{nw} \geq 0$: Number of nonwhite students assigned from neighborhood $k \in K$ to school $s \in S$.

Where:
- $S = \{\text{I}, \text{II}\}$ (schools)
- $K = \{\text{N01}, \text{N02}, \ldots, \text{N31}\}$ (neighborhoods)

##### Parameters

- School capacities:
  - $C_{\text{I}} = 2028$
  - $C_{\text{II}} = 1560$

- Neighborhood populations (white and nonwhite):

| Neighborhood | $P_k^w$ | $P_k^{nw}$ |
|--------------|---------|------------|
| N01          | 78      | 22         |
| N02          | 57      | 33         |
| N03          | 47      | 63         |
| N04          | 78      | 22         |
| N05          | 57      | 33         |
| N06          | 47      | 63         |
| N07          | 78      | 22         |
| N08          | 57      | 33         |
| N09          | 46      | 64         |
| N10          | 77      | 23         |
| N11          | 56      | 34         |
| N12          | 46      | 64         |
| N13          | 77      | 23         |
| N14          | 56      | 34         |
| N15          | 46      | 64         |
| N16          | 77      | 23         |
| N17          | 56      | 34         |
| N18          | 46      | 64         |
| N19          | 77      | 23         |
| N20          | 56      | 34         |
| N21          | 46      | 64         |
| N22          | 77      | 23         |
| N23          | 56      | 34         |
| N24          | 46      | 64         |
| N25          | 77      | 23         |
| N26          | 56      | 34         |
| N27          | 46      | 64         |
| N28          | 77      | 23         |
| N29          | 56      | 34         |
| N30          | 46      | 64         |
| N31          | 74      | 46         |

- Distances $d_{sk}$ (in miles) from school $s$ to neighborhood $k$:

| School | N01  | N02  | N03  | N04  | N05  | N06  | N07  | N08  | N09  | N10  | N11  | N12  | N13  | N14  | N15  | N16  | N17  | N18  | N19  | N20  | N21  | N22  | N23  | N24  | N25  | N26  | N27  | N28  | N29  | N30  | N31  |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| I      | 1.25 | 1.3  | 1.35 | 1.4  | 1.45 | 1.5  | 1.55 | 1.6  | 1.65 | 1.7  | 1.75 | 1.8  | 1.85 | 1.9  | 1.95 | 2.0  | 3.08 | 3.16 | 3.24 | 3.32 | 3.4  | 3.48 | 3.56 | 3.64 | 3.72 | 3.8  | 3.88 | 3.96 | 4.04 | 4.12 | 4.2  |
| II     | 3.08 | 3.16 | 3.24 | 3.32 | 3.4  | 3.48 | 3.56 | 3.64 | 3.72 | 3.8  | 3.88 | 3.96 | 4.04 | 4.12 | 4.2  | 4.28 | 1.25 | 1.3  | 1.35 | 1.4  | 1.45 | 1.5  | 1.55 | 1.6  | 1.65 | 1.7  | 1.75 | 1.8  | 1.85 | 1.9  | 1.95 |

- District-wide totals:
  - Total white students: $P^w = \sum_{k \in K} P_k^w = 1560$
  - Total nonwhite students: $P^{nw} = \sum_{k \in K} P_k^{nw} = 1040$
  - District white percentage: $r^w = \frac{1560}{1560+1040} = 0.6$
  - District nonwhite percentage: $r^{nw} = 0.4$

##### Objective Function

Minimize the total distance traveled by all students:
$$
\min \sum_{s \in S} \sum_{k \in K} d_{sk} \left( x_{sk}^w + x_{sk}^{nw} \right)
$$

##### Constraints

1. **Neighborhood assignment:** All students from each neighborhood must be assigned to some school.
   $$
   \sum_{s \in S} x_{sk}^w = P_k^w, \quad \forall k \in K
   $$
   $$
   \sum_{s \in S} x_{sk}^{nw} = P_k^{nw}, \quad \forall k \in K
   $$

2. **School capacity:** The total number of students assigned to each school cannot exceed its capacity.
   $$
   \sum_{k \in K} \left( x_{sk}^w + x_{sk}^{nw} \right) \leq C_s, \quad \forall s \in S
   $$

3. **Racial balance:** The percentage of white students at each school must be within 10 percentage points of the district ratio (i.e., between 50% and 70% white).
   $$
   0.5 \leq \frac{\sum_{k \in K} x_{sk}^w}{\sum_{k \in K} \left( x_{sk}^w + x_{sk}^{nw} \right)} \leq 0.7, \quad \forall s \in S
   $$
   (If the denominator is zero, the constraint is vacuously satisfied.)

4. **Nonnegativity:**
   $$
   x_{sk}^w \geq 0, \quad x_{sk}^{nw} \geq 0, \quad \forall s \in S, \forall k \in K
   $$

##### Summary of Sets and Parameters

- $S = \{\text{I}, \text{II}\}$
- $K = \{\text{N01}, \ldots, \text{N31}\}$
- $C_{\text{I}} = 2028$, $C_{\text{II}} = 1560$
- $P_k^w$, $P_k^{nw}$ as above
- $d_{sk}$ as above

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{s \in S} \sum_{k \in K} d_{sk} \left( x_{sk}^w + x_{sk}^{nw} \right) \\
\text{s.t.}\quad
& \sum_{s \in S} x_{sk}^w = P_k^w, && \forall k \in K \\
& \sum_{s \in S} x_{sk}^{nw} = P_k^{nw}, && \forall k \in K \\
& \sum_{k \in K} \left( x_{sk}^w + x_{sk}^{nw} \right) \leq C_s, && \forall s \in S \\
& 0.5 \leq \frac{\sum_{k \in K} x_{sk}^w}{\sum_{k \in K} \left( x_{sk}^w + x_{sk}^{nw} \right)} \leq 0.7, && \forall s \in S \\
& x_{sk}^w \geq 0,\ x_{sk}^{nw} \geq 0, && \forall s \in S,\, k \in K
\end{align*}
\]

All parameters and data are as retrieved above, with all vectors and matrices preserved.