Let $x_i$ be the number of units to produce of component $i$ ($i \in \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$), where $x_i \in \mathbb{Z}_{\geq 0}$.

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

The complete mathematical model is:

Objective:
\[
\max \sum_{i \in \{\text{C1}, \ldots, \text{C111}\}} p_i x_i
\]

Subject to (for each workshop $w$):

\[
\sum_{i \in \{\text{C1}, \ldots, \text{C111}\}} a_{wi} x_i \leq b_w \qquad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{\text{C1}, \ldots, \text{C111}\}
\]

Where the coefficients are as follows (all data preserved in source order):

#### Unit Processing Time $a_{wi}$ (from processing_time_unit.csv):

| Workshop         | C1   | C2   | C3   | ... | C111 |
|------------------|------|------|------|-----|------|
| Casting          | 0.74 | 0.77 | 1.41 | ... | 3.81 |
| Milling          | 0.6  | 3.38 | 0.0  | ... | 2.67 |
| Finishing        | 0.0  | 4.15 | 0.0  | ... | 4.07 |
| Assembly         | 4.84 | 0.0  | 3.8  | ... | 4.02 |
| QA & Packaging   | 0.92 | 0.0  | 1.08 | ... | 4.56 |

#### Unit Price $p_i$ (from unit_price.csv):

| Component | Unit Price |
|-----------|------------|
| C1        | 193        |
| C2        | 64         |
| C3        | 103        |
| ...       | ...        |
| C111      | 142        |

#### Total Working Hours $b_w$ (from total_working_hours.csv):

| Workshop         | Total Hours |
|------------------|-------------|
| Casting          | 7650        |
| Milling          | 6320        |
| Finishing        | 5538        |
| Assembly         | 5957        |
| QA & Packaging   | 6988        |

All coefficients and identifiers are as given in the retrieved data, in original file and row order.