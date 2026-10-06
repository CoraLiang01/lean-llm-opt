**Mathematical Model**

**Index Sets:**
- $I$: Set of products with SKU starting with ‘ZZ’ (from `RetailStoreSalesTransactions(ScannerData).csv`).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column `Revenue`).
- $d_i$: Demand for product $i$ (from column `Demand`).
- $s_i$: Initial Inventory of product $i$ (from column `Initial Inventory`).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:**  
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All rows in `RetailStoreSalesTransactions(ScannerData).csv` where `SKU` starts with "ZZ".
- $r_i$: `RetailStoreSalesTransactions(ScannerData).csv`, column `Revenue`, for each $i \in I$.
- $d_i$: `RetailStoreSalesTransactions(ScannerData).csv`, column `Demand`, for each $i \in I$.
- $s_i$: `RetailStoreSalesTransactions(ScannerData).csv`, column `Initial Inventory`, for each $i \in I$.