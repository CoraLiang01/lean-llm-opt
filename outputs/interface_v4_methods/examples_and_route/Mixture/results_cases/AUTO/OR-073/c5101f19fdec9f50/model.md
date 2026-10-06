#### Abstract Mathematical Model

Let:
- $P$ = set of products = {I, II, III}
- $E_A$ = set of equipment for procedure A = {A1, A2}
- $E_B$ = set of equipment for procedure B = {B1, B2, B3}
- $x_{p}$ = quantity of product $p$ produced (continuous, $\geq 0$)
- $y_{p,e}$ = quantity of product $p$ processed on equipment $e$ (continuous, $\geq 0$), for each relevant $(p,e)$ pair

Parameters (from 43.csv, see Data Mapping below):
- $t_{p,e}$ = processing time per unit of product $p$ on equipment $e$
- $T_e$ = available operating time for equipment $e$
- $C_e$ = equipment cost at full load for equipment $e$
- $c_p^{\text{raw}}$ = raw material cost per unit of product $p$
- $s_p$ = selling price per unit of product $p$

#### Decision Variables

- $x_p \geq 0$ (continuous): total production quantity of product $p$
- $y_{p,e} \geq 0$ (continuous): quantity of product $p$ processed on equipment $e$ (only for allowed $(p,e)$)

#### Objective Function

Maximize total profit:
$$
\max \left[ \sum_{p \in P} s_p x_p - \sum_{p \in P} c_p^{\text{raw}} x_p - \sum_{e} C_e \cdot \frac{1}{T_e} \sum_{p} t_{p,e} y_{p,e} \right]
$$

#### Constraints

1. **Production-Processing Consistency:**
   - For each product $p$ and each procedure (A and B), the sum of units processed on all allowed equipment for that procedure must equal $x_p$:
     - For procedure A:
       $$
       \sum_{e \in E_A(p)} y_{p,e} = x_p, \quad \forall p \in P
       $$
     - For procedure B:
       $$
       \sum_{e \in E_B(p)} y_{p,e} = x_p, \quad \forall p \in P
       $$
     - Where $E_A(p)$ and $E_B(p)$ are the sets of allowed equipment for product $p$ in procedures A and B, respectively (see Data Mapping).

2. **Equipment Capacity Constraints:**
   - For each equipment $e$:
     $$
     \sum_{p} t_{p,e} y_{p,e} \leq T_e
     $$

3. **Nonnegativity:**
   $$
   x_p \geq 0, \quad y_{p,e} \geq 0
   $$

#### Data Mapping

- **Product and Equipment Sets:**
  - $P$ = {I, II, III}
  - $E_A$ = {A1, A2}
  - $E_B$ = {B1, B2, B3}
- **Allowed Equipment per Product:**
  - $E_A$(I) = {A1, A2}; $E_B$(I) = {B1, B2, B3}
  - $E_A$(II) = {A1, A2}; $E_B$(II) = {B1}
  - $E_A$(III) = {A2}; $E_B$(III) = {B2}
- **Processing Times $t_{p,e}$:**
  - From 43.csv, table_id: file_0_view_0, columns: "Equipment / Cost", "Product I", "Product II", "Product III"
- **Available Equipment Operating Time $T_e$:**
  - From 43.csv, table_id: file_0_view_0, column: "Available Equipment Operating Time"
- **Equipment Cost at Full Load $C_e$:**
  - From 43.csv, table_id: file_0_view_0, column: "Equipment Cost at Full Load (yuan)"
- **Raw Material Cost $c_p^{\text{raw}}$ and Selling Price $s_p$:**
  - (Not present in the returned data; must be supplied from 43.csv if available elsewhere.)

#### Notes

- Only $(p,e)$ pairs with a non-empty processing time in 43.csv are allowed.
- All parameters must be mapped directly from the columns and rows of 43.csv as described above.
- All variables are continuous and nonnegative.
- The objective includes selling revenue, raw material cost, and equipment cost proportional to usage.

---

**All parameter values and allowed assignments must be mapped directly from 43.csv, table_id: file_0_view_0, using the original column and row names.**