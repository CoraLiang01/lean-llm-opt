**Sets**  
Let $\mathcal{I}$ be the set of all products classified under ‘ZZ’, indexed by $i$.

**Parameters**  
$A_i$ = Revenue per unit of product $i$ (from column “Revenue”, table_id: file_0_view_0)  
$d_i$ = Total demand for product $i$ (from column “Demand”, table_id: file_0_view_0)  
$I_i$ = Initial inventory of product $i$ (from column “Initial Inventory”, table_id: file_0_view_0)

**Decision Variables**  
$x_i$ = Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in \mathcal{I}$

**Objective**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i x_i
$$

**Constraints**  
1. Inventory constraint: $x_i \leq I_i \quad \forall i \in \mathcal{I}$
2. Demand constraint:  $x_i \leq d_i \quad \forall i \in \mathcal{I}$
3. Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

**Data Mapping**  
- Table: file_0_view_0 (from RetailStoreSalesTransactions(ScannerData).csv)
    - Set $\mathcal{I}$: All records where column “SKU” has prefix ‘ZZ’
    - Parameter $A_i$: column “Revenue”
    - Parameter $d_i$: column “Demand”
    - Parameter $I_i$: column “Initial Inventory”