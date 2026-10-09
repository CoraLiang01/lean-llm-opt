Sets:
- \( I \): Index set of products classified under ‘27in’. (From Product Name in file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (Revenue, file_0_view_0)
- \( d_i \): Demand for product \( i \). (Demand, file_0_view_0)
- \( s_i \): Initial inventory for product \( i \). (Initial Inventory, file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer, \( \forall i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Do not exceed demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Do not exceed initial inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Non-negativity)} \\
& x_i \in \mathbb{Z} && \forall i \in I \quad \text{(Integer units)}
\end{align*}
\]

Data Mapping:
- Set \( I \): All records in file_0_view_0 (SalesDataAnalysis.csv) where Product Name has prefix ‘27in’ (CSVQA-applied filter).
- \( r_i \): Revenue column, file_0_view_0.
- \( d_i \): Demand column, file_0_view_0.
- \( s_i \): Initial Inventory column, file_0_view_0.