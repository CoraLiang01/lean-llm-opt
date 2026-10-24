**Symbolic Model**

\[
\begin{aligned}
\text{Minimize}\quad & Z = \sum_{s\in S}\sum_{c\in C} c_{sc}\,x_{sc} \\[4pt]
\text{subject to}\quad
& \sum_{c\in C} x_{sc} \;\le\; s_s , && \forall s\in S \quad\text{(supply capacity)}\\[4pt]
& \sum_{s\in S} x_{sc} \;=\; d_c , && \forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
& x_{sc}\;\ge\;0 , && \forall s\in S,\;c\in C
\end{aligned}
\]

**Index Sets**

- \(S\) – set of Walmart stores (identified by the values in column **Unnamed: 0** of *supply_capacity.csv*).  
- \(C\) – set of customer groups (identified by the values in column **customer** of *customer_demand.csv*).

**Parameters**

| Symbol | Description | Source (table_id, column) |
|--------|-------------|---------------------------|
| \(d_c\) | Demand of customer \(c\) | `file_0_view_0`, column **demand** |
| \(s_s\) | Supply capacity of store \(s\) | `file_1_view_0`, column **supply_capacity** |
| \(c_{sc}\) | Transportation cost per unit from store \(s\) to customer \(c\) | `file_2_view_0`, columns **C1**–**C12** (matching customer identifiers) |

**Variables**

- \(x_{sc}\) – quantity shipped from store \(s\) to customer \(c\) (continuous, non‑negative).

**Data Mapping Rules**

1. **Customers**  
   - Extract the set \(C\) from the unique values in column **customer** of *customer_demand.csv* (`file_0_view_0`).  
   - Map each customer identifier to the corresponding demand value \(d_c\) from the same column.

2. **Stores**  
   - Extract the set \(S\) from the unique values in column **Unnamed: 0** of *supply_capacity.csv* (`file_1_view_0`).  
   - Map each store identifier to the corresponding supply capacity \(s_s\) from the same column.

3. **Transportation Costs**  
   - For each store row in *transportation_costs.csv* (`file_2_view_0`), the column **Unnamed: 0** gives the store identifier \(s\).  
   - Each remaining column **C1**–**C12** corresponds to a customer identifier \(c\); the cell value is the cost \(c_{sc}\).

All parameters are taken directly from the specified tables; no numerical values are listed in the model. The model is ready for implementation in any standard optimization solver.