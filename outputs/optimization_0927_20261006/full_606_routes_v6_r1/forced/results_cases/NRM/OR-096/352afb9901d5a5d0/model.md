#### Index Sets

- $S$: set of schools (from school_capacity.csv, column "School")
- $N$: set of neighborhoods (from neighborhoods_population.csv, column "Neighborhood")
- $R$: set of races, $R = \{\text{White}, \text{NonWhite}\}$

#### Parameters

- $C_s$: capacity of school $s \in S$ (from school_capacity.csv, column "Capacity")
- $P_{n,r}$: population of race $r \in R$ in neighborhood $n \in N$ (from neighborhoods_population.csv, columns "Population_White", "Population_NonWhite")
- $D_{s,n}$: distance in miles from school $s \in S$ to neighborhood $n \in N$ (from distance.csv, columns "School", $N$)
- $p^{\text{district}}_{\text{White}}$: district-wide proportion of white students (fixed at 0.6)
- $p^{\text{district}}_{\text{NonWhite}}$: district-wide proportion of nonwhite students (fixed at 0.4)
- $\delta$: allowed deviation from district racial proportion (fixed at 0.10)

#### Decision Variables

- $x_{s,n,r} \geq 0$: number of students of race $r \in R$ from neighborhood $n \in N$ assigned to school $s \in S$ (continuous or integer, as appropriate for population data)

#### Objective

Minimize total travel distance:
$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in R} D_{s,n} \cdot x_{s,n,r}
$$

#### Constraints

1. **Neighborhood Population Assignment**
   $$
   \sum_{s \in S} x_{s,n,r} = P_{n,r} \quad \forall n \in N, \; r \in R
   $$

2. **School Capacity**
   $$
   \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq C_s \quad \forall s \in S
   $$

3. **Racial Balance at Each School**
   $$
   p^{\text{district}}_{\text{White}} - \delta \leq \frac{\sum_{n \in N} x_{s,n,\text{White}}}{\sum_{n \in N} \sum_{r \in R} x_{s,n,r}} \leq p^{\text{district}}_{\text{White}} + \delta \quad \forall s \in S
   $$
   $$
   p^{\text{district}}_{\text{NonWhite}} - \delta \leq \frac{\sum_{n \in N} x_{s,n,\text{NonWhite}}}{\sum_{n \in N} \sum_{r \in R} x_{s,n,r}} \leq p^{\text{district}}_{\text{NonWhite}} + \delta \quad \forall s \in S
   $$
   (Only one of these is needed since the proportions sum to 1, but both are shown for clarity.)

4. **Nonnegativity**
   $$
   x_{s,n,r} \geq 0 \quad \forall s \in S, n \in N, r \in R
   $$

#### Data Mapping

- $S$, $C_s$: file_0_view_0 (school_capacity.csv), columns "School", "Capacity"
- $N$, $P_{n,\text{White}}$, $P_{n,\text{NonWhite}}$: file_1_view_0 (neighborhoods_population.csv), columns "Neighborhood", "Population_White", "Population_NonWhite"
- $D_{s,n}$: file_2_view_0 (distance.csv), row "School", columns $N$
- All rows and columns from each file are included, as returned by CSVQA. No additional filtering was applied.