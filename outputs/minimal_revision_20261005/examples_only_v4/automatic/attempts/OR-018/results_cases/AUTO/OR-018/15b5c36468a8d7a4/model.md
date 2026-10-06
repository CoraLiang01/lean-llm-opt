Mathematical Optimization Model

Sets:
- \( I \): Set of products classified under ‘Baby’. (Index \( i \in I \))
  (Source: all records in file_0_view_0, column Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Source: file_0_view_0, column Revenue)
- \( d_i \): Demand for product \( i \).
  (Source: file_0_view_0, column Demand)
- \( s_i \): Initial inventory of product \( i \).
  (Source: file_0_view_0, column Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill.
  (Domain: integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \))

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \\
& x_i \leq s_i && \forall i \in I \\
& x_i \geq 0 && \forall i \in I \\
& x_i \in \mathbb{Z} && \forall i \in I
\end{align*}
\]

Data Mapping

- Set \( I \): file_0_view_0, column Product Name (all returned records)
- Parameter \( r_i \): file_0_view_0, column Revenue
- Parameter \( d_i \): file_0_view_0, column Demand
- Parameter \( s_i \): file_0_view_0, column Initial Inventory