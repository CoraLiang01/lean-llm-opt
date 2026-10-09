---

### Abstract Mathematical Model

#### Index Sets
- $I$ : Set of products classified under ‘id999’.

#### Parameters
- $A_i$ : Revenue per unit of product $i \in I$.
- $d_i$ : Demand for product $i \in I$ during the sales horizon.
- $I_i$ : Initial inventory of product $i \in I$.

#### Decision Variables
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$  
  Number of units of product $i$ to fulfill (non-negative integer).

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraints**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraints**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

### Data Mapping

- **Source Table:** `file_0_view_0` (from `OnlineRetailSalesDataset.csv`)
- **Selection Rule:** $I = \{ i : \text{row } i \text{ has } \texttt{id_number} = \text{'id999'} \}$
- **Parameter Columns:**
    - $A_i$ : `Revenue`
    - $d_i$ : `Demand`
    - $I_i$ : `Initial Inventory`

---

**End of Model.**