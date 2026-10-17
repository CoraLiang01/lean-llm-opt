#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products in the source table whose "Product Name" contains the substring "FAUX".

**Parameters:**
- $a_i$: Revenue per unit of product $i \in I$ (from column "Revenue").
- $d_i$: Total demand for product $i \in I$ (from column "Demand").
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `ZARASales.csv`)
- **Subset Selection:** $I = \{i : \text{row } i \text{ in } \texttt{file\_0\_view\_0} \text{ with "Product Name" containing "FAUX"}\}$
- **Parameter Columns:**
    - $a_i$: "Revenue"
    - $d_i$: "Demand"
    - $s_i$: "Initial Inventory"
- **Decision Variable:** $x_i$ for each $i \in I$