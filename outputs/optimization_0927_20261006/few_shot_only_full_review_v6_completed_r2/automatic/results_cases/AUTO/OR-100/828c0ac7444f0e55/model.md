Let $x_k$ be the number of units to produce of component $k$ ($k \in \{\text{C1}, \ldots, \text{C111}\}$).

**Parameters:**
- $p_k$: unit price of component $k$ (from unit_price.csv)
- $a_{wk}$: processing time required in workshop $w$ for one unit of component $k$ (from processing_time_unit.csv)
- $b_w$: total available working hours in workshop $w$ (from total_working_hours.csv)
- Workshops $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$

**Decision variables:**
- $x_k \in \mathbb{Z}_{\geq 0}$, for all $k \in \{\text{C1}, \ldots, \text{C111}\}$

**Objective:**
\[
\max \sum_{k=\text{C1}}^{\text{C111}} p_k x_k
\]

**Subject to:**

For each workshop $w$:
\[
\sum_{k=\text{C1}}^{\text{C111}} a_{wk} x_k \leq b_w
\]
where:
- For Casting: $b_{\text{Casting}} = 7650$
- For Milling: $b_{\text{Milling}} = 6320$
- For Finishing: $b_{\text{Finishing}} = 5538$
- For Assembly: $b_{\text{Assembly}} = 5957$
- For QA & Packaging: $b_{\text{QA \& Packaging}} = 6988$

**Variable domains:**
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k \in \{\text{C1}, \ldots, \text{C111}\}
\]

**Numerical Formulation (with all coefficients):**

Let $k$ index the 111 components, with the following mapping (from the CSVs):

| $k$ | $p_k$ | $a_{\text{Casting},k}$ | $a_{\text{Milling},k}$ | $a_{\text{Finishing},k}$ | $a_{\text{Assembly},k}$ | $a_{\text{QA \& Packaging},k}$ |
|-----|-------|------------------------|------------------------|--------------------------|-------------------------|-------------------------------|
| C1  | 193   | 0.74                   | 0.60                   | 0.00                     | 4.84                    | 0.92                          |
| C2  | 64    | 0.77                   | 3.38                   | 4.15                     | 0.00                    | 0.00                          |
| ... | ...   | ...                    | ...                    | ...                      | ...                     | ...                           |
| C111| 142   | 3.81                   | 2.67                   | 4.07                     | 4.02                    | 3.21                          |

**Objective:**
\[
\max \left(
193x_{\text{C1}} + 64x_{\text{C2}} + 103x_{\text{C3}} + \cdots + 142x_{\text{C111}}
\right)
\]

**Subject to:**
\[
\begin{align*}
&0.74x_{\text{C1}} + 0.77x_{\text{C2}} + \cdots + 3.81x_{\text{C111}} \leq 7650 \quad \text{(Casting)} \\
&0.60x_{\text{C1}} + 3.38x_{\text{C2}} + \cdots + 2.67x_{\text{C111}} \leq 6320 \quad \text{(Milling)} \\
&0.00x_{\text{C1}} + 4.15x_{\text{C2}} + \cdots + 4.07x_{\text{C111}} \leq 5538 \quad \text{(Finishing)} \\
&4.84x_{\text{C1}} + 0.00x_{\text{C2}} + \cdots + 4.02x_{\text{C111}} \leq 5957 \quad \text{(Assembly)} \\
&0.92x_{\text{C1}} + 0.00x_{\text{C2}} + \cdots + 3.21x_{\text{C111}} \leq 6988 \quad \text{(QA \& Packaging)} \\
&x_k \in \mathbb{Z}_{\geq 0} \quad \forall k \in \{\text{C1}, \ldots, \text{C111}\}
\end{align*}
\]

All coefficients $p_k$ and $a_{wk}$ are as given in the CSVs above, in original order.

**Summary:**
- Maximize total output value.
- Each workshop's total time used by all produced components cannot exceed its available hours.
- All production quantities are nonnegative integers.