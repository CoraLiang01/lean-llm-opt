#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products classified under ‘ZZ’ in the dataset.

**Parameters:**
- $r_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in the source file).
- $d_i$: Deterministic demand for product $i \in I$ (from column ‘Demand’).
- $s_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
2. **Inventory availability:** 
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
3. **Nonnegativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

**Data Mapping:**

- **Source file:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv`
- **Index set $I$:** All rows where the product is classified under ‘ZZ’ (as specified in the user query).
- **Parameter $r_i$:** Column `Revenue`.
- **Parameter $d_i$:** Column `Demand`.
- **Parameter $s_i$:** Column `Initial Inventory`.

**Note:** All columns and rows are included as returned by the CSVQA tool; no further selection or filtering is performed beyond the explicit subset for ‘ZZ’ products as specified in the user query.