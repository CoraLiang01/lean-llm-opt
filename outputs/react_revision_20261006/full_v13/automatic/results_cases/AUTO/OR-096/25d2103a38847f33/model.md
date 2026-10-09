## Mathematical Model

### Sets
- $S$: set of schools (from file_0_view_0, column "School")
- $N$: set of neighborhoods (from file_1_view_0, column "Neighborhood")

### Parameters
- $C_s$: capacity of school $s \in S$ (from file_0_view_0, column "Capacity")
- $W_n$: number of white students in neighborhood $n \in N$ (from file_1_view_0, column "Population_White")
- $B_n$: number of nonwhite students in neighborhood $n \in N$ (from file_1_view_0, column "Population_NonWhite")
- $d_{sn}$: distance in miles from school $s$ to neighborhood $n$ (from file_2_view_0, row "School" $s$, column $n$)

- $W_{tot} = \sum_{n \in N} W_n$: total white students in the district
- $B_{tot} = \sum_{n \in N} B_n$: total nonwhite students in the district
- $P_{white} = \frac{W_{tot}}{W_{tot} + B_{tot}}$: district-wide white percentage (should be 0.6 per problem statement)
- $P_{nonwhite} = 1 - P_{white}$

### Decision Variables
- $x_{sn}^W \geq 0$: number of white students from neighborhood $n$ assigned to school $s$
- $x_{sn}^B \geq 0$: number of nonwhite students from neighborhood $n$ assigned to school $s$

### Objective
Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} d_{sn} \left( x_{sn}^W + x_{sn}^B \right)
$$

### Constraints

1. **Neighborhood assignment (all students assigned):**
   $$
   \sum_{s \in S} x_{sn}^W = W_n \qquad \forall n \in N
   $$
   $$
   \sum_{s \in S} x_{sn}^B = B_n \qquad \forall n \in N
   $$

2. **School capacity:**
   $$
   \sum_{n \in N} \left( x_{sn}^W + x_{sn}^B \right) \leq C_s \qquad \forall s \in S
   $$

3. **Racial balance at each school:**
   $$
   P_{white} - 0.10 \leq \frac{\sum_{n \in N} x_{sn}^W}{\sum_{n \in N} (x_{sn}^W + x_{sn}^B)} \leq P_{white} + 0.10 \qquad \forall s \in S
   $$
   (If $\sum_{n \in N} (x_{sn}^W + x_{sn}^B) > 0$; if a school is empty, the ratio is undefined but the constraint is vacuously satisfied.)

4. **Nonnegativity:**
   $$
   x_{sn}^W \geq 0, \quad x_{sn}^B \geq 0 \qquad \forall s \in S, n \in N
   $$

---

### Data Mapping

- $S$: file_0_view_0, column "School"
- $N$: file_1_view_0, column "Neighborhood"
- $C_s$: file_0_view_0, column "Capacity", key "School"
- $W_n$: file_1_view_0, column "Population_White", key "Neighborhood"
- $B_n$: file_1_view_0, column "Population_NonWhite", key "Neighborhood"
- $d_{sn}$: file_2_view_0, row "School" $s$, column $n$
- $x_{sn}^W$, $x_{sn}^B$: decision variables for assignment of white/nonwhite students from $n$ to $s$

- $P_{white}$: computed as $\frac{\sum_{n \in N} W_n}{\sum_{n \in N} (W_n + B_n)}$ from file_1_view_0

---

**All sets, parameters, and constraints are mapped directly to the current CSV data as described above.**