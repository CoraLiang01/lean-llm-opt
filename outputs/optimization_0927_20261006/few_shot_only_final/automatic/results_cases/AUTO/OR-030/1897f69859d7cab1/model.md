---

**Sets:**  
- $I$: Set of all car models with identifier ‘FDK57’ (from table_id: file_0_view_0, column: Product Name).

**Parameters:**  
- $A_i$: Revenue per unit for car model $i \in I$ (from table_id: file_0_view_0, column: Revenue).
- $d_i$: Demand for car model $i \in I$ (from table_id: file_0_view_0, column: Demand).
- $I_i$: Initial inventory for car model $i \in I$ (from table_id: file_0_view_0, column: Initial Inventory).

**Decision Variables:**  
- $x_i$: Quantity of car model $i \in I$ to fulfill, $x_i \geq 0$.

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq I_i, \quad \forall i \in I \\
& x_i \geq 0, \quad \forall i \in I
\end{align*}
\]

---

**Data Mapping:**  
- All parameters and index sets are drawn from table_id: file_0_view_0 (source: BigMartSales.csv), using columns:
    - Product Name (for set $I$ and filtering on ‘FDK57’)
    - Revenue (for $A_i$)
    - Demand (for $d_i$)
    - Initial Inventory (for $I_i$)

---