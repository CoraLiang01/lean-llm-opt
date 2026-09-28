#### Index Sets

- $S$: set of schools (from school_capacity.csv, column "School")
- $N$: set of neighborhoods (from neighborhoods_population.csv, column "Neighborhood")
- $R$: set of races, $R = \{\text{White}, \text{NonWhite}\}$

#### Parameters

- $C_s$: capacity of school $s \in S$ (school_capacity.csv, column "Capacity")
- $P_{n,r}$: population of race $r \in R$ in neighborhood $n \in N$ (neighborhoods_population.csv, columns "Population_White", "Population_NonWhite")
- $D_{s,n}$: distance in miles from school $s \in S$ to neighborhood $n \in N$ (distance.csv, columns "School", $n$)
- $p^{\text{district}}_{\text{White}} = 0.6$, $p^{\text{district}}_{\text{NonWhite}} = 0.4$ (district-wide racial proportions, from question)
- $\delta = 0.10$ (allowed deviation in racial balance, from question)

#### Decision Variables

- $x_{s,n,r} \geq 0$: number of students of race $r \in R$ from neighborhood $n \in N$ assigned to school $s \in S$ (continuous or integer, as appropriate for population data)

#### Objective

Minimize total travel distance:
$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in R} D_{s,n} \cdot x_{s,n,r}
$$

#### Constraints

1. **Neighborhood Assignment (all students assigned):**
   $$
   \sum_{s \in S} x_{s,n,r} = P_{n,r} \qquad \forall n \in N, \; r \in R
   $$

2. **School Capacity:**
   $$
   \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq C_s \qquad \forall s \in S
   $$

3. **Racial Balance at Each School:**
   $$
   p^{\text{district}}_{\text{White}} - \delta \leq \frac{\sum_{n \in N} x_{s,n,\text{White}}}{\sum_{n \in N} \sum_{r \in R} x_{s,n,r}} \leq p^{\text{district}}_{\text{White}} + \delta \qquad \forall s \in S
   $$
   $$
   p^{\text{district}}_{\text{NonWhite}} - \delta \leq \frac{\sum_{n \in N} x_{s,n,\text{NonWhite}}}{\sum_{n \in N} \sum_{r \in R} x_{s,n,r}} \leq p^{\text{district}}_{\text{NonWhite}} + \delta \qquad \forall s \in S
   $$
   (Only one of these is needed since the percentages sum to 1; typically, the White percentage is used.)

   Equivalently, for White:
   $$
   (p^{\text{district}}_{\text{White}} - \delta) \cdot \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq \sum_{n \in N} x_{s,n,\text{White}} \leq (p^{\text{district}}_{\text{White}} + \delta) \cdot \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \qquad \forall s \in S
   $$

4. **Nonnegativity:**
   $$
   x_{s,n,r} \geq 0 \qquad \forall s \in S, \; n \in N, \; r \in R
   $$

   (If populations are integer, add $x_{s,n,r} \in \mathbb{Z}_+$.)

---

#### Data Mapping

- **school_capacity.csv**: 
  - Table ID: file_0_view_0
  - Columns: "School" $\rightarrow S$, "Capacity" $\rightarrow C_s$
- **neighborhoods_population.csv**: 
  - Table ID: file_1_view_0
  - Columns: "Neighborhood" $\rightarrow N$, "Population_White" $\rightarrow P_{n,\text{White}}$, "Population_NonWhite" $\rightarrow P_{n,\text{NonWhite}}$
- **distance.csv**: 
  - Table ID: file_2_view_0
  - Columns: "School" $\rightarrow S$, each neighborhood column $\rightarrow D_{s,n}$

District-wide racial proportions and allowed deviation are specified in the user query.