Let $x_i$ be the number of units produced of component $i$ ($i = \text{C1}, \text{C2}, \ldots, \text{C111}$), where $x_i \in \mathbb{Z}_{\geq 0}$.

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time (in hours) required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in$ {Casting, Milling, Finishing, Assembly, QA & Packaging}).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

The complete mathematical model is:

Objective:
\[
\max \sum_{i=\text{C1}}^{\text{C111}} p_i x_i
\]

Subject to (for each workshop $w$):

\[
\sum_{i=\text{C1}}^{\text{C111}} a_{wi} x_i \leq b_w \qquad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = \text{C1}, \ldots, \text{C111}
\]

Where the coefficients are as follows (all values as retrieved, in source order):

#### Processing times $a_{wi}$ (hours per unit):

| Workshop           | C1   | C2   | C3   | ... | C111 |
|--------------------|------|------|------|-----|------|
| Casting            | 0.74 | 0.77 | 1.41 | ... | 3.81 |
| Milling            | 0.6  | 3.38 | 0.0  | ... | 2.67 |
| Finishing          | 0.0  | 4.15 | 0.0  | ... | 4.07 |
| Assembly           | 4.84 | 0.0  | 3.8  | ... | 4.02 |
| QA & Packaging     | 0.92 | 0.0  | 1.08 | ... | 3.21 |

#### Unit prices $p_i$:

| Component | Unit Price |
|-----------|------------|
| C1        | 193        |
| C2        | 64         |
| C3        | 103        |
| ...       | ...        |
| C111      | 142        |

#### Workshop total available hours $b_w$:

| Workshop         | Total Hours |
|------------------|-------------|
| Casting          | 7650        |
| Milling          | 6320        |
| Finishing        | 5538        |
| Assembly         | 5957        |
| QA & Packaging   | 6988        |

All coefficients and identifiers are as retrieved and in original order. The model maximizes total output value, subject to the available working hours in each workshop, with integer nonnegative production quantities for each component.