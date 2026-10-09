Let \( x_j \) be the number of units to produce of component \( j \) (for \( j = 1, \ldots, 111 \)), where each component is identified by its code (C1, C2, ..., C111). All \( x_j \) are nonnegative integers.

Define:
- Let \( p_j \) be the unit price of component \( j \) (from unit_price.csv).
- Let \( t_{i,j} \) be the unit processing time (in hours) required for component \( j \) in workshop \( i \) (from processing_time_unit.csv), where \( i \) is one of: Casting, Milling, Finishing, Assembly, QA & Packaging.
- Let \( H_i \) be the total available working hours in workshop \( i \) (from total_working_hours.csv).

The model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{j=1}^{111} p_j x_j \\
\text{subject to} \quad
& \sum_{j=1}^{111} t_{i,j} x_j \leq H_i \quad \forall i \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\} \\
& x_j \geq 0 \text{ and integer} \quad \forall j = 1, \ldots, 111
\end{align*}
\]

With explicit coefficients from the data:

Let the set of components be \( J = \{\text{C1}, \text{C2}, ..., \text{C111}\} \).

Variables:
- \( x_j \in \mathbb{Z}_+ \) for all \( j \in J \): number of units to produce of component \( j \).

Objective:
\[
\text{Maximize} \quad 193x_{\text{C1}} + 64x_{\text{C2}} + 103x_{\text{C3}} + \cdots + 142x_{\text{C111}}
\]
(where the coefficients are the unit prices from unit_price.csv, in the order given.)

Constraints:

1. Casting capacity:
\[
0.74x_{\text{C1}} + 0.77x_{\text{C2}} + 1.41x_{\text{C3}} + \cdots + 3.81x_{\text{C111}} \leq 7650
\]

2. Milling capacity:
\[
0.6x_{\text{C1}} + 3.38x_{\text{C2}} + 0.0x_{\text{C3}} + \cdots + 2.67x_{\text{C111}} \leq 6320
\]

3. Finishing capacity:
\[
0.0x_{\text{C1}} + 4.15x_{\text{C2}} + 0.0x_{\text{C3}} + \cdots + 4.07x_{\text{C111}} \leq 5538
\]

4. Assembly capacity:
\[
4.84x_{\text{C1}} + 0.0x_{\text{C2}} + 3.8x_{\text{C3}} + \cdots + 4.02x_{\text{C111}} \leq 5957
\]

5. QA & Packaging capacity:
\[
0.92x_{\text{C1}} + 0.0x_{\text{C2}} + 1.08x_{\text{C3}} + \cdots + 3.21x_{\text{C111}} \leq 6988
\]

Where all coefficients are taken directly from the corresponding rows and columns of processing_time_unit.csv, and the right-hand sides from total_working_hours.csv.

Domain:
\[
x_j \in \mathbb{Z}_+, \quad \forall j \in J
\]

This model arranges production to maximize total output value, subject to the working hour limits of each workshop, using all provided data and identifiers.