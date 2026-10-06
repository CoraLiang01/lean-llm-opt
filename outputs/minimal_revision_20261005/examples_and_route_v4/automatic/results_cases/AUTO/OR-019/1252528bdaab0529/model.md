**Abstract Mathematical Model**

**Index Set:**
- $I$: Set of products classified under ‘27in’ (from all rows in `SalesDataAnalysis.csv` where `Product Name` starts with "27in").

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (`Revenue`, from `SalesDataAnalysis.csv`).
- $d_i$: Demand for product $i$ (`Demand`, from `SalesDataAnalysis.csv`).
- $s_i$: Initial inventory of product $i$ (`Initial Inventory`, from `SalesDataAnalysis.csv`).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   $\forall i \in I: \quad x_i \leq d_i$
2. **Inventory limit:**  
   $\forall i \in I: \quad x_i \leq s_i$
3. **Nonnegativity and integrality:**  
   $\forall i \in I: \quad x_i \in \mathbb{Z}_{\geq 0}$

---

**Data Mapping**

- $I$: All records in `SalesDataAnalysis.csv` where `Product Name` starts with "27in".
- $r_i$: `SalesDataAnalysis.csv`, column `Revenue`, for each $i \in I$.
- $d_i$: `SalesDataAnalysis.csv`, column `Demand`, for each $i \in I$.
- $s_i$: `SalesDataAnalysis.csv`, column `Initial Inventory`, for each $i \in I$.

**Variable Mapping**

- $x_i$: Number of units of product $i$ to fulfill, for each $i \in I$.