Let $x_i$ be the number of units to produce of component $i$ ($i \in \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$).

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

The model is:

$$
\max \sum_{i \in \{\text{C1}, \ldots, \text{C111}\}} p_i x_i
$$

Subject to, for each workshop $w$:
$$
\sum_{i \in \{\text{C1}, \ldots, \text{C111}\}} a_{wi} x_i \leq b_w
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{\text{C1}, \ldots, \text{C111}\}
$$

Where:

- $p_i$ is given by the "unit_price" column in unit_price.csv for each component $i$.
- $a_{wi}$ is given by the value in row $w$, column $i$ of processing_time_unit.csv.
- $b_w$ is given by the "total_hours" column in total_working_hours.csv for each workshop $w$.

Explicitly, the constraints are:

- $\sum_{i} a_{\text{Casting},i} x_i \leq 7650$
- $\sum_{i} a_{\text{Milling},i} x_i \leq 6320$
- $\sum_{i} a_{\text{Finishing},i} x_i \leq 5538$
- $\sum_{i} a_{\text{Assembly},i} x_i \leq 5957$
- $\sum_{i} a_{\text{QA \& Packaging},i} x_i \leq 6988$

And $x_i$ are nonnegative integers for all $i \in \{\text{C1}, \ldots, \text{C111}\}$.