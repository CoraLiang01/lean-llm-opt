##### Decision Variables

$x_i \geq 0$: scale of development per day in area $i \in A$ (continuous).

##### Parameters

Let $A$ be the set of areas:
\[
A = \{\text{Queens},\ \text{Brooklyn},\ \text{Manhattan},\ \text{Bronx},\ \text{Staten Island},\ \text{Harlem},\ \text{Upper East Side},\ \text{Lower Manhattan},\ \text{Midtown},\ \text{Long Island City},\ \text{Williamsburg},\ \text{Bushwick},\ \text{Flatbush},\ \text{Greenpoint},\ \text{Park Slope},\ \text{Astoria},\ \text{Jackson Heights},\ \text{Flushing},\ \text{Sunnyside},\ \text{Ditmars}\}
\]

Benefit per unit developed in each area $i$ ($v_i$):

\[
\begin{aligned}
&v_{\text{Queens}} = 469,\quad w_{\text{Queens}} = 954 \\
&v_{\text{Brooklyn}} = 290,\quad w_{\text{Brooklyn}} = 650 \\
&v_{\text{Manhattan}} = 236,\quad w_{\text{Manhattan}} = 961 \\
&v_{\text{Bronx}} = 235,\quad w_{\text{Bronx}} = 950 \\
&v_{\text{Staten Island}} = 745,\quad w_{\text{Staten Island}} = 379 \\
&v_{\text{Harlem}} = 684,\quad w_{\text{Harlem}} = 776 \\
&v_{\text{Upper East Side}} = 444,\quad w_{\text{Upper East Side}} = 381 \\
&v_{\text{Lower Manhattan}} = 172,\quad w_{\text{Lower Manhattan}} = 808 \\
&v_{\text{Midtown}} = 1000,\quad w_{\text{Midtown}} = 937 \\
&v_{\text{Long Island City}} = 336,\quad w_{\text{Long Island City}} = 608 \\
&v_{\text{Williamsburg}} = 546,\quad w_{\text{Williamsburg}} = 912 \\
&v_{\text{Bushwick}} = 535,\quad w_{\text{Bushwick}} = 391 \\
&v_{\text{Flatbush}} = 539,\quad w_{\text{Flatbush}} = 465 \\
&v_{\text{Greenpoint}} = 831,\quad w_{\text{Greenpoint}} = 490 \\
&v_{\text{Park Slope}} = 139,\quad w_{\text{Park Slope}} = 918 \\
&v_{\text{Astoria}} = 432,\quad w_{\text{Astoria}} = 787 \\
&v_{\text{Jackson Heights}} = 627,\quad w_{\text{Jackson Heights}} = 347 \\
&v_{\text{Flushing}} = 629,\quad w_{\text{Flushing}} = 274 \\
&v_{\text{Sunnyside}} = 292,\quad w_{\text{Sunnyside}} = 642 \\
&v_{\text{Ditmars}} = 978,\quad w_{\text{Ditmars}} = 130 \\
\end{aligned}
\]

Total development capacity: $C = 586$

##### Objective Function

\[
\max \sum_{i \in A} v_i x_i
\]

##### Constraints

1. Capacity constraint:
   \[
   \sum_{i \in A} w_i x_i \leq C
   \]
2. Nonnegativity:
   \[
   x_i \geq 0 \quad \forall i \in A
   \]

##### Retrieved Information

- Areas $A$:
  - Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars
- Benefit per unit $v_i$ and weight per unit $w_i$ for each area as listed above.
- Total development capacity $C = 586$.

This is a continuous knapsack-type maximization model for optimal real estate development allocation.