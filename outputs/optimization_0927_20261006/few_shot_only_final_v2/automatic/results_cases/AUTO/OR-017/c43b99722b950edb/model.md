**Sets:**
- $I$: Set of all products classified under 'ZZ'.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column 'Revenue').
- $d_i$: Demand for product $i \in I$ (from column 'Demand').
- $I_i$: Initial inventory for product $i \in I$ (from column 'Initial Inventory').

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory constraint:**
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
2. **Demand constraint:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

**Data Mapping:**

- Source Table: `file_0_view_0` (from `RetailStoreSalesTransactions(ScannerData).csv`)
- Product Set $I$: All rows where the product is classified under 'ZZ' (as defined by the user or dataset).
- $A_i$: Column `Revenue`
- $d_i$: Column `Demand`
- $I_i$: Column `Initial Inventory`