##### Decision Variables

Let $x_{s,n,w} \geq 0$ = number of white students assigned from neighborhood $n$ to school $s$ (continuous)  
Let $x_{s,n,b} \geq 0$ = number of nonwhite students assigned from neighborhood $n$ to school $s$ (continuous)

where  
$s \in S$ = set of schools (from `file_0_view_0`, column `School`)  
$n \in N$ = set of neighborhoods (from `file_1_view_0`, column `Neighborhood`)  
$r \in R = \{\text{w}, \text{b}\}$ = race, with w = white, b = nonwhite

##### Parameters

- $C_s$ = capacity of school $s$ (from `file_0_view_0`, column `Capacity`)
- $P_{n,w}$ = white population in neighborhood $n$ (from `file_1_view_0`, column `Population_White`)
- $P_{n,b}$ = nonwhite population in neighborhood $n$ (from `file_1_view_0`, column `Population_NonWhite`)
- $d_{s,n}$ = distance in miles from school $s$ to neighborhood $n$ (from `file_2_view_0`, columns `N01`–`N31`, rows indexed by `School`)
- $W_{\text{district}} = \sum_{n \in N} P_{n,w}$ = total white students in district
- $B_{\text{district}} = \sum_{n \in N} P_{n,b}$ = total nonwhite students in district
- $T_{\text{district}} = W_{\text{district}} + B_{\text{district}}$ = total students in district
- $\rho = 0.6$ = district white percentage (from query)
- $\epsilon = 0.10$ = allowed deviation in white percentage

##### Objective

Minimize total student-miles traveled:
$$
\min \sum_{s \in S} \sum_{n \in N} d_{s,n} \left( x_{s,n,w} + x_{s,n,b} \right)
$$

##### Constraints

1. **Neighborhood assignment:**  
   All students from each neighborhood must be assigned:
   $$
   \sum_{s \in S} x_{s,n,w} = P_{n,w} \quad \forall n \in N
   $$
   $$
   \sum_{s \in S} x_{s,n,b} = P_{n,b} \quad \forall n \in N
   $$

2. **School capacity:**  
   School enrollment cannot exceed capacity:
   $$
   \sum_{n \in N} \left( x_{s,n,w} + x_{s,n,b} \right) \leq C_s \quad \forall s \in S
   $$

3. **Racial balance at each school:**  
   The white percentage at each school must be within $[\rho - \epsilon, \rho + \epsilon]$:
   $$
   \rho - \epsilon \leq \frac{\sum_{n \in N} x_{s,n,w}}{\sum_{n \in N} (x_{s,n,w} + x_{s,n,b})} \leq \rho + \epsilon \quad \forall s \in S
   $$
   For linearity, rewrite as:
   $$
   \sum_{n \in N} x_{s,n,w} \geq (\rho - \epsilon) \sum_{n \in N} (x_{s,n,w} + x_{s,n,b}) \quad \forall s \in S
   $$
   $$
   \sum_{n \in N} x_{s,n,w} \leq (\rho + \epsilon) \sum_{n \in N} (x_{s,n,w} + x_{s,n,b}) \quad \forall s \in S
   $$

4. **Nonnegativity:**  
   $$
   x_{s,n,w} \geq 0,\quad x_{s,n,b} \geq 0 \quad \forall s \in S, n \in N
   $$

---

#### Data Mapping

- $S$ = all values in `file_0_view_0`, column `School`
- $N$ = all values in `file_1_view_0`, column `Neighborhood`
- $C_s$ = `file_0_view_0`, column `Capacity`, row where `School` = $s$
- $P_{n,w}$ = `file_1_view_0`, column `Population_White`, row where `Neighborhood` = $n$
- $P_{n,b}$ = `file_1_view_0`, column `Population_NonWhite`, row where `Neighborhood` = $n$
- $d_{s,n}$ = `file_2_view_0`, column $n$, row where `School` = $s$
- $W_{\text{district}} = \sum_{n \in N} P_{n,w}$
- $B_{\text{district}} = \sum_{n \in N} P_{n,b}$
- $T_{\text{district}} = W_{\text{district}} + B_{\text{district}}$
- $\rho = 0.6$, $\epsilon = 0.10$ (from query)

---

**Index sets, parameters, and all coefficients are bound exactly to the retrieved data as described above.**