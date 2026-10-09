Let $x_k$ = number of units of component $k$ to produce, $k \in \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$.

Let $p_k$ = unit price of component $k$ (from unit_price.csv).

Let $a_{wk}$ = unit processing time of component $k$ in workshop $w$ (from processing_time_unit.csv), for $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$.

Let $b_w$ = total available working hours in workshop $w$ (from total_working_hours.csv).

**Objective:**
\[
\max \sum_{k=1}^{111} p_k x_k
\]

**Constraints:**

For each workshop $w$:
\[
\sum_{k=1}^{111} a_{wk} x_k \leq b_w
\]
where
- For Casting: $b_{\text{Casting}} = 7650$
- For Milling: $b_{\text{Milling}} = 6320$
- For Finishing: $b_{\text{Finishing}} = 5538$
- For Assembly: $b_{\text{Assembly}} = 5957$
- For QA & Packaging: $b_{\text{QA \& Packaging}} = 6988$

**Variable domains:**
\[
x_k \in \mathbb{Z}_{\geq 0}, \quad \forall k \in \{\text{C1}, \ldots, \text{C111}\}
\]

**Numerical Data:**

- For $k$ from C1 to C111:
    - $p_k$ from unit_price.csv (e.g., $p_{\text{C1}} = 193$, $p_{\text{C2}} = 64$, ..., $p_{\text{C111}} = 142$)
    - $a_{wk}$ from processing_time_unit.csv (e.g., $a_{\text{Casting},\text{C1}} = 0.74$, $a_{\text{Milling},\text{C1}} = 0.6$, ..., $a_{\text{QA \& Packaging},\text{C111}} = 3.21$)

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{k=1}^{111} p_k x_k \\
\text{s.t.} \quad
& \sum_{k=1}^{111} a_{\text{Casting},k} x_k \leq 7650 \\
& \sum_{k=1}^{111} a_{\text{Milling},k} x_k \leq 6320 \\
& \sum_{k=1}^{111} a_{\text{Finishing},k} x_k \leq 5538 \\
& \sum_{k=1}^{111} a_{\text{Assembly},k} x_k \leq 5957 \\
& \sum_{k=1}^{111} a_{\text{QA \& Packaging},k} x_k \leq 6988 \\
& x_k \in \mathbb{Z}_{\geq 0} \quad \forall k \in \{\text{C1}, \ldots, \text{C111}\}
\end{align*}
\]

Where all coefficients $p_k$ and $a_{wk}$ are as given in the source CSVs, and all constraints and variables are as above.