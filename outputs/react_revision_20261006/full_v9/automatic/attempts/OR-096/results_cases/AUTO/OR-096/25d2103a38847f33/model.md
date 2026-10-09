## Symbolic Mathematical Model

### Sets
- $S$: set of schools (from school_capacity.csv), indexed by $s$
- $N$: set of neighborhoods (from neighborhoods_population.csv and distance.csv), indexed by $n$
- $R$: set of races, $R = \{\text{White}, \text{NonWhite}\}$, indexed by $r$

### Parameters
- $C_s$: capacity of school $s$ (from school_capacity.csv; table_id: file_0_view_0, column: Capacity)
- $P_{n,r}$: number of students of race $r$ in neighborhood $n$ (from neighborhoods_population.csv; table_id: file_1_view_0, columns: Population_White, Population_NonWhite)
- $d_{s,n}$: distance in miles from school $s$ to neighborhood $n$ (from distance.csv; table_id: file_2_view_0, columns: School, N01...N31)
- $P^{\text{total}}_r = \sum_{n \in N} P_{n,r}$: total number of students of race $r$ in the district
- $P^{\text{total}} = \sum_{r \in R} P^{\text{total}}_r$: total number of students in the district
- $\alpha_r$: district-wide proportion of race $r$ students, $\alpha_r = P^{\text{total}}_r / P^{\text{total}}$
- $\epsilon = 0.10$: allowed deviation in white-student percentage (10 percentage points)

### Decision Variables
- $x_{s,n,r} \geq 0$: number of students of race $r$ from neighborhood $n$ assigned to school $s$

### Objective
Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in R} d_{s,n} \cdot x_{s,n,r}
$$

### Constraints

1. **Neighborhood population assignment:**  
   All students of each race from each neighborhood must be assigned to some school:
   $$
   \sum_{s \in S} x_{s,n,r} = P_{n,r} \qquad \forall n \in N, \; r \in R
   $$

2. **School capacity:**  
   The total number of students assigned to each school does not exceed its capacity:
   $$
   \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq C_s \qquad \forall s \in S
   $$

3. **Racial balance at each school:**  
   The percentage of white students at each school must be within 10 percentage points of the district-wide percentage (60% white, 40% nonwhite):
   $$
   \alpha_{\text{White}} - \epsilon \leq 
   \frac{\sum_{n \in N} x_{s,n,\text{White}}}{\sum_{n \in N} \sum_{r \in R} x_{s,n,r}}
   \leq \alpha_{\text{White}} + \epsilon
   \qquad \forall s \in S
   $$
   (If $\sum_{n \in N} \sum_{r \in R} x_{s,n,r} = 0$, the fraction is undefined, but this cannot occur if all students are assigned.)

   With $\alpha_{\text{White}} = 0.6$, $\epsilon = 0.1$, so $0.5 \leq \text{white \% at school} \leq 0.7$.

   Equivalently, for all $s \in S$:
   $$
   0.5 \cdot \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq \sum_{n \in N} x_{s,n,\text{White}} \leq 0.7 \cdot \sum_{n \in N} \sum_{r \in R} x_{s,n,r}
   $$

4. **Nonnegativity:**
   $$
   x_{s,n,r} \geq 0 \qquad \forall s \in S, \; n \in N, \; r \in R
   $$

---

### Data Mapping

- $S$: All schools in file_0_view_0 (school_capacity.csv), column "School"
- $N$: All neighborhoods in file_1_view_0 (neighborhoods_population.csv), column "Neighborhood"
- $R$: $\{\text{White}, \text{NonWhite}\}$
- $C_s$: file_0_view_0, column "Capacity", key "School"
- $P_{n,\text{White}}$: file_1_view_0, column "Population_White", key "Neighborhood"
- $P_{n,\text{NonWhite}}$: file_1_view_0, column "Population_NonWhite", key "Neighborhood"
- $d_{s,n}$: file_2_view_0 (distance.csv), row "School" = $s$, column $n$
- $\alpha_{\text{White}} = \frac{\sum_{n} P_{n,\text{White}}}{\sum_{n} (P_{n,\text{White}} + P_{n,\text{NonWhite}})}$
- $\alpha_{\text{NonWhite}} = 1 - \alpha_{\text{White}}$

---

**All sets, parameters, and constraints are defined directly from the provided CSVs and the problem statement.**