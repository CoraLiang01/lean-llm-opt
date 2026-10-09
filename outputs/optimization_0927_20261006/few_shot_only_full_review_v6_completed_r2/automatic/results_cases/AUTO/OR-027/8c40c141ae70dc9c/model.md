---

### Abstract Mathematical Model

#### Index Sets
- $I$ : Set of all products in the dataset classified as ‘Organ’ in the column **Sub Category**.

#### Parameters
- $A_i$ : Revenue per unit of product $i \in I$, from **Revenue** column.
- $d_i$ : Total demand for product $i \in I$, from **Demand** column.
- $I_i$ : Initial inventory for product $i \in I$, from **Initial Inventory** column.

#### Decision Variables
- $x_i$ : Number of units of product $i \in I$ to fulfill (integer, $x_i \geq 0$).

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_{+} \qquad \forall i \in I
   \]

---

### Data Mapping

- **Source Table:**  
  - `file_0_view_0` (from `SupermartGrocerySales-RetailAnalyticsDataset.csv`)
- **Index Set Selection:**  
  - $I$ is the set of all records where `Sub Category` contains the substring ‘Organ’.
- **Parameter Columns:**  
  - $A_i$: `Revenue`
  - $d_i$: `Demand`
  - $I_i$: `Initial Inventory`

---

**Note:**  
- All parameters and index sets are defined symbolically and mapped to the exact columns and table as returned in the Observation.
- The selection of ‘Organ’ products is implemented by filtering records where `Sub Category` contains ‘Organ’.