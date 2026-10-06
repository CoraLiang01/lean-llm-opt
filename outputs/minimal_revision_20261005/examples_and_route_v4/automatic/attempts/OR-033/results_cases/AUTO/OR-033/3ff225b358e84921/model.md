**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products classified under ‘Baby’ (from EuropeSalesRecords.csv, "Product Name" with prefix "Baby")

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (EuropeSalesRecords.csv, "Revenue")
- $d_i$: Demand for product $i$ (EuropeSalesRecords.csv, "Demand")
- $s_i$: Initial inventory of product $i$ (EuropeSalesRecords.csv, "Initial Inventory")

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory constraint:** 
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

| Symbol | Source Table/Column | Mapping Rule |
|--------|--------------------|--------------|
| $I$    | EuropeSalesRecords.csv, "Product Name" | All rows where "Product Name" starts with "Baby" |
| $r_i$  | EuropeSalesRecords.csv, "Revenue" | For each $i \in I$, use "Revenue" |
| $d_i$  | EuropeSalesRecords.csv, "Demand" | For each $i \in I$, use "Demand" |
| $s_i$  | EuropeSalesRecords.csv, "Initial Inventory" | For each $i \in I$, use "Initial Inventory" |

**Variable Domain:**  
$x_i$ is a nonnegative integer for each $i \in I$.