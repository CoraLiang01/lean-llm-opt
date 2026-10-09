#### Symbolic Model

**Index Sets:**
- $I$: Set of all products with SKU starting with 'ZZ' (i.e., all 'ZZ' products).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column 'Revenue').
- $d_i$: Total demand for product $i \in I$ (from column 'Demand').
- $I_i$: Initial inventory for product $i \in I$ (from column 'Initial Inventory').

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
2. **Inventory limit:** 
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- Source Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv`
- Index Set $I$: All rows where `SKU` starts with 'ZZ'
- Parameter $A_i$: Column `Revenue`
- Parameter $d_i$: Column `Demand`
- Parameter $I_i$: Column `Initial Inventory`