---

### Abstract Optimization Model

#### Index Sets
- $I$: Set of car models with `Product Name` = 'FDK57' (from the data table).

#### Parameters
- $A_i$: Revenue per unit for model $i \in I$ (from column `Revenue`).
- $d_i$: Demand for model $i \in I$ (from column `Demand`).
- $I_i$: Initial inventory for model $i \in I$ (from column `Initial Inventory`).

#### Decision Variables
- $x_i \in \mathbb{Z}_+$: Number of units of model $i \in I$ to fulfill (integer, $x_i \geq 0$).

#### Objective
\[
\max \sum_{i \in I} A_i x_i
\]

#### Constraints
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \\
& x_i \leq I_i, && \forall i \in I \\
& x_i \in \mathbb{Z}_+, && \forall i \in I
\end{align*}
\]

---

### Data Mapping

- **Source Table**: `file_0_view_0` (from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv`)
- **Selection**: All records where `Product Name` = 'FDK57'
- **Parameter Columns**:
    - $A_i$: `Revenue`
    - $d_i$: `Demand`
    - $I_i$: `Initial Inventory`

---

**End of Model**