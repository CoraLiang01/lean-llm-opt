Let:
- \( x_j \) = number of units to produce of component \( j \), for \( j = 1, \ldots, 111 \), where each component is identified by its explicit ID \( Cj \) (e.g., \( C1, C2, \ldots, C111 \)).
- All \( x_j \) are nonnegative integers: \( x_j \in \mathbb{Z}_+, \forall j \).

Parameters:
- Let \( p_j \) = unit price of component \( j \) (from unit_price.csv).
- Let \( t_{i,j} \) = unit processing time (in hours) required for component \( j \) in workshop \( i \) (from processing_time_unit.csv), where \( i \) is one of: Casting, Milling, Finishing, Assembly, QA & Packaging.
- Let \( T_i \) = total available working hours in workshop \( i \) (from total_working_hours.csv).

Data (all values as per the original row order and file order):

Workshops and their total available hours:
- Casting: \( T_{\text{Casting}} = 7650 \)
- Milling: \( T_{\text{Milling}} = 6320 \)
- Finishing: \( T_{\text{Finishing}} = 5538 \)
- Assembly: \( T_{\text{Assembly}} = 5957 \)
- QA & Packaging: \( T_{\text{QA\&Packaging}} = 6988 \)

For each component \( j \) (from \( C1 \) to \( C111 \)), the unit price \( p_j \) and the unit processing times \( t_{i,j} \) for each workshop \( i \) are as given in the CSVs.

Model:

\[
\begin{align*}
\text{Maximize} \quad & Z = \sum_{j=1}^{111} p_j x_j \\
\text{subject to} \quad
& \sum_{j=1}^{111} t_{\text{Casting},j} x_j \leq 7650 \\
& \sum_{j=1}^{111} t_{\text{Milling},j} x_j \leq 6320 \\
& \sum_{j=1}^{111} t_{\text{Finishing},j} x_j \leq 5538 \\
& \sum_{j=1}^{111} t_{\text{Assembly},j} x_j \leq 5957 \\
& \sum_{j=1}^{111} t_{\text{QA\&Packaging},j} x_j \leq 6988 \\
& x_j \in \mathbb{Z}_+, \quad \forall j = 1, \ldots, 111
\end{align*}
\]

Where:
- \( p_j \) is the unit price for component \( Cj \) (from unit_price.csv, in the supplied order).
- \( t_{i,j} \) is the unit processing time for component \( Cj \) in workshop \( i \) (from processing_time_unit.csv, in the supplied order).
- All coefficients and bounds are taken directly from the CSVs, preserving their order.

This model arranges production to maximize total output value, subject to the available working hours in each workshop, using all provided data and respecting all explicit keys and boundaries.