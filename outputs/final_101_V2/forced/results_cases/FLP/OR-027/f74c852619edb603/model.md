##### Decision Variables

Let:
- $x_1$: Number of units of Organic Fruits fulfilled
- $x_2$: Number of units of Organic Staples fulfilled
- $x_3$: Number of units of Organic Vegetables fulfilled

All $x_i \geq 0$ and integer (if partial units are not allowed; otherwise, continuous).

##### Parameters

| Product             | Revenue per unit | Initial Inventory | Demand  |
|---------------------|------------------|------------------|---------|
| Organic Fruits      | 60.8             | 5,034,020        | 678,906 |
| Organic Staples     | 918.45           | 5,589,290        | 749,927 |
| Organic Vegetables  | 77.52            | 5,202,710        | 699,808 |

##### Objective Function

\[
\max\ 60.8\,x_1 + 918.45\,x_2 + 77.52\,x_3
\]

##### Constraints

1. Inventory and demand limits for each product:
   \[
   0 \leq x_1 \leq \min(5,\!034,\!020,\ 678,\!906)
   \]
   \[
   0 \leq x_2 \leq \min(5,\!589,\!290,\ 749,\!927)
   \]
   \[
   0 \leq x_3 \leq \min(5,\!202,\!710,\ 699,\!808)
   \]

Or, equivalently:
   \[
   0 \leq x_1 \leq 678,\!906
   \]
   \[
   0 \leq x_2 \leq 749,\!927
   \]
   \[
   0 \leq x_3 \leq 699,\!808
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\max\quad & 60.8\,x_1 + 918.45\,x_2 + 77.52\,x_3 \\
\text{s.t.}\quad
& 0 \leq x_1 \leq 678,\!906 \\
& 0 \leq x_2 \leq 749,\!927 \\
& 0 \leq x_3 \leq 699,\!808 \\
& x_1, x_2, x_3 \geq 0
\end{align*}
\]

where:
- $x_1$: units of Organic Fruits fulfilled
- $x_2$: units of Organic Staples fulfilled
- $x_3$: units of Organic Vegetables fulfilled

All parameters are as retrieved from the data.