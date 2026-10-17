#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all car models classified under ‘FDK57’ (as identified by the ‘Product Name’ column in the data source).

**Parameters:**
- $A_i$: Revenue per unit for car model $i \in I$ (from column ‘Revenue’).
- $d_i$: Deterministic demand for car model $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for car model $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Quantity of car model $i \in I$ to fulfill, $x_i \geq 0$ (real, non-negative).

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
2. **Inventory constraint:**
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
3. **Non-negativity:**
   \[
   x_i \geq 0 \qquad \forall i \in I
   \]

---

**Data Mapping:**

- **Source file:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv`
- **Index set $I$:** All rows where `Product Name` is classified under ‘FDK57’ (i.e., all car models with `Product Name` = ‘FDK57’).
- **Parameter $A_i$:** Value from column `Revenue` for each $i \in I$.
- **Parameter $d_i$:** Value from column `Demand` for each $i \in I$.
- **Parameter $I_i$:** Value from column `Initial Inventory` for each $i \in I$.

---

**Notes:**
- No additional eligibility, selection, or activation constraints are imposed by the query.
- The variable domain is non-negative real unless otherwise specified in the query.
- All data columns and rows are included as returned by the CSVQA tool; subset selection for ‘FDK57’ is explicit in the model definition above.