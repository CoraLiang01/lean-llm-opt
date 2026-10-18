#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of drug types (indexed by $i$)

**Parameters:**
- $v_i$: benefit coefficient for drug type $i$ (from products.csv, column "Value")
- $w_i$: weight per unit for drug type $i$ (from products.csv, column "Weight")
- $C$: overall inventory capacity (from capacity.csv, column "Capacity")

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of drug type $i$ to order daily

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
1. **Overall Capacity Constraint:**
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$, $v_i$, $w_i$ are defined by all records and columns "ProductName", "Value", and "Weight" in table_id: file_1_view_0 (products.csv).
- $C$ is defined by column "Capacity" in table_id: file_0_view_0 (capacity.csv).
- No additional filters or subsets were applied; all records from both tables are used.