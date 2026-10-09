#### Abstract Mathematical Model

**Index Sets:**
- $P$: set of all products (from 41.csv, column "Product Name")

**Parameters:**
- $a_p$: labor requirement per unit of product $p$ (from 41.csv, column "Labor per unit"), $\forall p \in P$
- $b_p$: material requirement per unit of product $p$ (from 41.csv, column "Material per unit"), $\forall p \in P$
- $s_p$: selling price per unit of product $p$ (from 41.csv, column "Selling Price"), $\forall p \in P$
- $c_p$: variable cost per unit of product $p$ (from 41.csv, column "Variable Cost"), $\forall p \in P$
- $L$: total weekly labor capacity (given, $L = 1650$)
- $M$: total weekly material capacity (given, $M = 1850$)
- $F$: fixed weekly operating cost (given, $F = 4500$)

**Decision Variables:**
- $x_p$: production quantity of product $p$ (continuous, $x_p \geq 0$, $\forall p \in P$)

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
   x_p \geq 0, \quad \forall p \in P
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
- **No filters applied:** All product records used. 

- **External Parameters:** $L$, $M$, $F$ are given in the user query, not in the table.