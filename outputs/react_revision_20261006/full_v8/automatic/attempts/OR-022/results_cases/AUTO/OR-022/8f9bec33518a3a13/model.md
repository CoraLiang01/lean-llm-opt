##### Mathematical Optimization Model

Let $I$ be the set of all products whose "Product Name" contains "27in".

**Index Set:**
- $I$ : set of all products classified under ‘27in’ (i.e., all products in the data whose "Product Name" contains "27in")

**Parameters:**
- $A_i$ : revenue per unit of product $i \in I$ (from column "Revenue")
- $d_i$ : deterministic demand for product $i \in I$ (from column "Demand")
- $s_i$ : initial inventory for product $i \in I$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$ : number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

##### Data Mapping

- Table: file_0_view_0 (from Salesorders.csv)
    - Index set $I$: All rows where "Product Name" contains "27in"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"