## Sets
- $S$: set of schools (from file_0_view_0, column School)
- $N$: set of neighborhoods (from file_1_view_0, column Neighborhood)
- $C$: set of student categories, $C = \{\text{White}, \text{NonWhite}\}$

## Parameters
- $\text{Cap}_s$: capacity of school $s \in S$ (file_0_view_0, Capacity)
- $\text{Pop}_{n,c}$: number of students of category $c \in C$ in neighborhood $n \in N$ (file_1_view_0, Population_White and Population_NonWhite)
- $\text{Dist}_{s,n}$: distance in miles from school $s$ to neighborhood $n$ (file_2_view_0, columns N01...N31, rows indexed by School)
- $\alpha = 0.6$: district-wide white student proportion
- $\beta = 0.4$: district-wide nonwhite student proportion
- $\delta = 0.1$: allowed deviation in white percentage

## Decision Variables
- $x_{s,n,c} \geq 0$: number of students of category $c$ from neighborhood $n$ assigned to school $s$

## Objective
Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{c \in C} \text{Dist}_{s,n} \cdot x_{s,n,c}
$$

## Constraints

**1. All students assigned:**
$$
\sum_{s \in S} x_{s,n,c} = \text{Pop}_{n,c} \quad \forall n \in N,\, c \in C
$$

**2. School capacity:**
$$
\sum_{n \in N} \sum_{c \in C} x_{s,n,c} \leq \text{Cap}_s \quad \forall s \in S
$$

**3. Racial balance at each school:**
Let $W_s = \sum_{n \in N} x_{s,n,\text{White}}$, $T_s = \sum_{n \in N} \sum_{c \in C} x_{s,n,c}$

$$
(\alpha - \delta) \cdot T_s \leq W_s \leq (\alpha + \delta) \cdot T_s \quad \forall s \in S
$$

**4. Non-negativity:**
$$
x_{s,n,c} \geq 0 \quad \forall s \in S,\, n \in N,\, c \in C
$$

---

## Data Mapping

- $S$: file_0_view_0, column School
- $N$: file_1_view_0, column Neighborhood
- $C$: $\{\text{White}, \text{NonWhite}\}$
- $\text{Cap}_s$: file_0_view_0, column Capacity, indexed by School
- $\text{Pop}_{n,\text{White}}$: file_1_view_0, column Population_White, indexed by Neighborhood
- $\text{Pop}_{n,\text{NonWhite}}$: file_1_view_0, column Population_NonWhite, indexed by Neighborhood
- $\text{Dist}_{s,n}$: file_2_view_0, row School $s$, column $n$
- $\alpha = 0.6$, $\beta = 0.4$, $\delta = 0.1$ (from problem statement)

---

**Indices, parameters, and all constraints are mapped directly to the current CSV data as described above.**