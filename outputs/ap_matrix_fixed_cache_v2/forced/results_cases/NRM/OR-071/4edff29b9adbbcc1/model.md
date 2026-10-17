#### Abstract Mathematical Model

**Index Sets:**
- $P$: Set of all products (from 41.csv, column "Product Name").

**Parameters:**
- $a_p$: Labor required per unit of product $p \in P$ (from "Labor per unit").
- $b_p$: Material required per unit of product $p \in P$ (from "Material per unit").
- $s_p$: Selling price per unit of product $p \in P$ (from "Selling Price").
- $c_p$: Variable cost per unit of product $p \in P$ (from "Variable Cost").
- $L$: Total available labor per week (given as 1,650).
- $M$: Total available material per week (given as 1,850).
- $F$: Fixed weekly operating cost (given as 4,500).

**Decision Variables:**
- $x_p \geq 0$: Continuous production quantity of product $p \in P$.

**Objective:**
\[
\max \left( \sum_{p \in P} (s_p - c_p) x_p - F \right)
\]

**Constraints:**
1. **Labor Capacity:**
   \[
   \sum_{p \in P} a_p x_p \leq L
   \]
2. **Material Capacity:**
   \[
   \sum_{p \in P} b_p x_p \leq M
   \]
3. **Nonnegativity:**
   \[
   x_p \geq 0 \quad \forall p \in P
   \]

---

#### Data Mapping

- **Table:** 41.csv
- **table_id:** file_0_view_0
- **Columns Used:**
  - Product Name $\rightarrow$ $P$
  - Labor per unit $\rightarrow$ $a_p$
  - Material per unit $\rightarrow$ $b_p$
  - Selling Price $\rightarrow$ $s_p$
  - Variable Cost $\rightarrow$ $c_p$
- **All records from file_0_view_0 are included (no filter).**
- **Aggregate parameters $L$, $M$, $F$ are from the user query.**