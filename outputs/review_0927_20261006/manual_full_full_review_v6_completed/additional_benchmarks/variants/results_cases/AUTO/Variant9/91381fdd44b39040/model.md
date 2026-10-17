Let $y_p$ be the integer number of standard rolls cut using pattern $p$.

**Parameters:**

- Items: $I = \{\text{A}, \text{B}, \text{C}, \text{D}\}$
- Patterns: $P = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}, \text{P7}, \text{P8}, \text{P9}\}$
- Demand for each item:
  - $d_{\text{A}} = 24$
  - $d_{\text{B}} = 18$
  - $d_{\text{C}} = 12$
  - $d_{\text{D}} = 10$
- Units of item $i$ produced by pattern $p$: $a_{ip}$, given by:

| Pattern | A | B | C | D |
|---------|---|---|---|---|
| P1      | 4 | 0 | 0 | 0 |
| P2      | 0 | 3 | 0 | 0 |
| P3      | 0 | 0 | 2 | 0 |
| P4      | 0 | 0 | 0 | 2 |
| P5      | 2 | 1 | 0 | 0 |
| P6      | 1 | 0 | 1 | 0 |
| P7      | 0 | 1 | 0 | 1 |
| P8      | 1 | 1 | 1 | 0 |
| P9      | 2 | 0 | 0 | 1 |

**Decision Variables:**

- $y_p \in \mathbb{Z}_{\geq 0}$, for all $p \in P$

**Objective:**

Minimize the total number of standard rolls used:
$$
\min \sum_{p \in P} y_p
$$

**Constraints:**

For each item $i \in I$, the total number of units produced must meet or exceed demand:
\[
\begin{align*}
&\sum_{p \in P} a_{\text{A},p} \, y_p \geq 24 \\
&\sum_{p \in P} a_{\text{B},p} \, y_p \geq 18 \\
&\sum_{p \in P} a_{\text{C},p} \, y_p \geq 12 \\
&\sum_{p \in P} a_{\text{D},p} \, y_p \geq 10 \\
\end{align*}
\]
where the $a_{ip}$ coefficients are as given in the table above.

**Variable Domains:**
\[
y_p \in \mathbb{Z}_{\geq 0} \quad \forall p \in P
\]

**Complete Model:**

Minimize:
\[
y_{\text{P1}} + y_{\text{P2}} + y_{\text{P3}} + y_{\text{P4}} + y_{\text{P5}} + y_{\text{P6}} + y_{\text{P7}} + y_{\text{P8}} + y_{\text{P9}}
\]

Subject to:
\[
\begin{align*}
4y_{\text{P1}} + 0y_{\text{P2}} + 0y_{\text{P3}} + 0y_{\text{P4}} + 2y_{\text{P5}} + 1y_{\text{P6}} + 0y_{\text{P7}} + 1y_{\text{P8}} + 2y_{\text{P9}} &\geq 24 \\
0y_{\text{P1}} + 3y_{\text{P2}} + 0y_{\text{P3}} + 0y_{\text{P4}} + 1y_{\text{P5}} + 0y_{\text{P6}} + 1y_{\text{P7}} + 1y_{\text{P8}} + 0y_{\text{P9}} &\geq 18 \\
0y_{\text{P1}} + 0y_{\text{P2}} + 2y_{\text{P3}} + 0y_{\text{P4}} + 0y_{\text{P5}} + 1y_{\text{P6}} + 0y_{\text{P7}} + 1y_{\text{P8}} + 0y_{\text{P9}} &\geq 12 \\
0y_{\text{P1}} + 0y_{\text{P2}} + 0y_{\text{P3}} + 2y_{\text{P4}} + 0y_{\text{P5}} + 0y_{\text{P6}} + 1y_{\text{P7}} + 0y_{\text{P8}} + 1y_{\text{P9}} &\geq 10 \\
\end{align*}
\]

\[
y_{\text{P1}}, y_{\text{P2}}, y_{\text{P3}}, y_{\text{P4}}, y_{\text{P5}}, y_{\text{P6}}, y_{\text{P7}}, y_{\text{P8}}, y_{\text{P9}} \in \mathbb{Z}_{\geq 0}
\]