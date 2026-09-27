### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products classified as ‘27in’.

#### Parameters
- $A_i$: Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

### Data Mapping

- Table ID: file_0_view_0
- Source File: SalesDataAnalysis.csv
- Columns Used:
    - Product Name (filtered for prefix ‘27in’)
    - Revenue
    - Demand
    - Initial Inventory

All parameters $A_i$, $d_i$, and $I_i$ are mapped from the respective columns for each product $i$ in the set of ‘27in’ products.