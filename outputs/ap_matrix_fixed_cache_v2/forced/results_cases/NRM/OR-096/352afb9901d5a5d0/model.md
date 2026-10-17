#### Sets
- $S$: set of schools (from `school_capacity.csv`, column `School`)
- $N$: set of neighborhoods (from `neighborhoods_population.csv`, column `Neighborhood`)
- $R$: set of races, $R = \{\text{White}, \text{NonWhite}\}$

#### Parameters
- $C_s$: capacity of school $s \in S$ (from `school_capacity.csv`, column `Capacity`)
- $P_{n,r}$: population of race $r \in R$ in neighborhood $n \in N$ (from `neighborhoods_population.csv`, columns `Population_White`, `Population_NonWhite`)
- $D_{s,n}$: distance in miles from school $s \in S$ to neighborhood $n \in N$ (from `distance.csv`, columns indexed by `School` and `N01`–`N31`)
- $\alpha$: district-wide white student proportion (fixed at $0.6$)
- $\beta$: allowed deviation from district proportion (fixed at $0.1$)

#### Decision Variables
- $x_{s,n,r} \geq 0$: number of students of race $r \in R$ from neighborhood $n \in N$ assigned to school $s \in S$

#### Objective
Minimize total travel distance:
$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in R} D_{s,n} \cdot x_{s,n,r}
$$

#### Constraints

1. **Neighborhood Assignment:**  
   All students of each race from each neighborhood must be assigned to some school:
   $$
   \sum_{s \in S} x_{s,n,r} = P_{n,r} \quad \forall n \in N, \; r \in R
   $$

2. **School Capacity:**  
   The total number of students assigned to each school cannot exceed its capacity:
   $$
   \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq C_s \quad \forall s \in S
   $$

3. **Racial Balance at Each School:**  
   The proportion of white students at each school must be within $[\alpha - \beta, \alpha + \beta]$:
   $$
   (\alpha - \beta) \cdot \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq \sum_{n \in N} x_{s,n,\text{White}} \leq (\alpha + \beta) \cdot \sum_{n \in N} \sum_{r \in R} x_{s,n,r} \quad \forall s \in S
   $$

4. **Nonnegativity and Integrality:**  
   $$
   x_{s,n,r} \geq 0 \text{ and integer} \quad \forall s \in S, n \in N, r \in R
   $$

---

#### Data Mapping

- **school_capacity.csv** (`file_0_view_0`):  
  - Set $S$ from column `School`
  - Parameter $C_s$ from column `Capacity`
- **neighborhoods_population.csv** (`file_1_view_0`):  
  - Set $N$ from column `Neighborhood`
  - Parameters $P_{n,\text{White}}$ from column `Population_White`, $P_{n,\text{NonWhite}}$ from column `Population_NonWhite`
- **distance.csv** (`file_2_view_0`):  
  - Parameter $D_{s,n}$ from value at row with `School` $s$ and column $n$ (`N01`–`N31`)

District-wide white proportion $\alpha$ and deviation $\beta$ are fixed scenario parameters from the query, not from the data files.