##### Decision Variables

Let $x_{s,n}^W \geq 0$ = number of white students assigned from neighborhood $n$ to school $s$ (continuous)  
Let $x_{s,n}^N \geq 0$ = number of nonwhite students assigned from neighborhood $n$ to school $s$ (continuous)  

where $s \in S$ (schools), $n \in N$ (neighborhoods), $S = \{\text{I}, \text{II}\}$, $N = \{\text{N01}, \ldots, \text{N31}\}$.

##### Parameters

- $C_s$: capacity of school $s$ (from school_capacity.csv)
- $P_n^W$: white population in neighborhood $n$ (from neighborhoods_population.csv)
- $P_n^N$: nonwhite population in neighborhood $n$ (from neighborhoods_population.csv)
- $d_{s,n}$: distance in miles from school $s$ to neighborhood $n$ (from distance.csv)
- $P^W = \sum_{n \in N} P_n^W$: total white students in the district
- $P^N = \sum_{n \in N} P_n^N$: total nonwhite students in the district
- $r^W = \frac{P^W}{P^W + P^N}$: district-wide white percentage (should be 0.6 per description, but calculated from data)
- $r^N = 1 - r^W$: district-wide nonwhite percentage

##### Objective

Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} d_{s,n} \left( x_{s,n}^W + x_{s,n}^N \right)
$$

##### Constraints

1. **Neighborhood assignment (all students assigned):**
   $$
   \sum_{s \in S} x_{s,n}^W = P_n^W \quad \forall n \in N
   $$
   $$
   \sum_{s \in S} x_{s,n}^N = P_n^N \quad \forall n \in N
   $$

2. **School capacity:**
   $$
   \sum_{n \in N} \left( x_{s,n}^W + x_{s,n}^N \right) \leq C_s \quad \forall s \in S
   $$

3. **Racial balance at each school:**
   $$
   r^W - 0.10 \leq \frac{\sum_{n \in N} x_{s,n}^W}{\sum_{n \in N} (x_{s,n}^W + x_{s,n}^N)} \leq r^W + 0.10 \quad \forall s \in S
   $$
   (If $\sum_{n \in N} (x_{s,n}^W + x_{s,n}^N) > 0$; if a school is empty, the ratio is undefined but the constraint is vacuously satisfied.)

4. **Nonnegativity:**
   $$
   x_{s,n}^W \geq 0,\quad x_{s,n}^N \geq 0 \quad \forall s \in S, n \in N
   $$

##### Data Mapping

- $S$: Schools from school_capacity.csv, column "School", table_id file_0_view_0
- $C_s$: Capacity from school_capacity.csv, column "Capacity", table_id file_0_view_0
- $N$: Neighborhoods from neighborhoods_population.csv, column "Neighborhood", table_id file_1_view_0
- $P_n^W$: Population_White from neighborhoods_population.csv, column "Population_White", table_id file_1_view_0
- $P_n^N$: Population_NonWhite from neighborhoods_population.csv, column "Population_NonWhite", table_id file_1_view_0
- $d_{s,n}$: Distance from distance.csv, row "School" = $s$, column $n$, table_id file_2_view_0
- $r^W$: Computed as $\frac{\sum_{n \in N} P_n^W}{\sum_{n \in N} (P_n^W + P_n^N)}$ from file_1_view_0

All index sets, parameters, and coefficients are defined by the retrieved data.