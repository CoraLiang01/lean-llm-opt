##### Sets

- $I = \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$: component types
- $W = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$: workshops

##### Parameters

- $p_i$: unit price of component $i$
- $a_{wi}$: unit processing time (hours) of component $i$ in workshop $w$
- $T_w$: total available working hours in workshop $w$

##### Decision Variables

- $x_i \geq 0$: production quantity of component $i$ (continuous)

##### Objective

\[
\max \sum_{i \in I} p_i x_i
\]

##### Constraints

For each workshop $w \in W$:
\[
\sum_{i \in I} a_{wi} x_i \leq T_w
\]

For all $i \in I$:
\[
x_i \geq 0
\]

---

#### Numerical Model

Let $I = \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$ and $W = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$.

**Parameters:**

- Unit prices $p_i$:

| Component | Unit Price |
|-----------|------------|
| C1        | 193        |
| C2        | 64         |
| C3        | 103        |
| C4        | 210        |
| C5        | 85         |
| C6        | 126        |
| C7        | 226        |
| C8        | 94         |
| C9        | 73         |
| C10       | 120        |
| ...       | ...        |
| C111      | 142        |

- Unit processing times $a_{wi}$ (hours per unit):

| Workshop           | C1   | C2   | ... | C111 |
|--------------------|------|------|-----|------|
| Casting            | 0.74 | 0.77 | ... | 3.81 |
| Milling            | 0.6  | 3.38 | ... | 2.67 |
| Finishing          | 0.0  | 4.15 | ... | 4.07 |
| Assembly           | 4.84 | 0.0  | ... | 4.02 |
| QA & Packaging     | 0.92 | 0.0  | ... | 3.21 |

- Total available working hours $T_w$:

| Workshop           | Total Hours |
|--------------------|-------------|
| Casting            | 7650        |
| Milling            | 6320        |
| Finishing          | 5538        |
| Assembly           | 5957        |
| QA & Packaging     | 6988        |

**Model:**

\[
\max \sum_{i=1}^{111} p_i x_i
\]

Subject to:

\[
\sum_{i=1}^{111} a_{\text{Casting},i} x_i \leq 7650
\]
\[
\sum_{i=1}^{111} a_{\text{Milling},i} x_i \leq 6320
\]
\[
\sum_{i=1}^{111} a_{\text{Finishing},i} x_i \leq 5538
\]
\[
\sum_{i=1}^{111} a_{\text{Assembly},i} x_i \leq 5957
\]
\[
\sum_{i=1}^{111} a_{\text{QA \& Packaging},i} x_i \leq 6988
\]
\[
x_i \geq 0 \quad \forall i = 1, \ldots, 111
\]

Where all $p_i$ and $a_{wi}$ are as given in the retrieved data above, preserving all identifiers and coefficients.