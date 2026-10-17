Let $I$ be the set of drug types, indexed by $i$, with names and coefficients as in the table below.

Let $x_i$ = number of units of drug type $i$ to order daily (integer, $\geq 0$).

Parameters (from products.csv):

\[
\begin{array}{lll}
\text{ProductName} & \text{Value } (v_i) & \text{Weight } (w_i) \\
\hline
\text{NSAIDs} & 585 & 50 \\
\text{Antirheumatic Drugs} & 557 & 329 \\
\text{Acetic Acid Derivatives} & 963 & 410 \\
\text{Antibiotics} & 301 & 452 \\
\text{Antiviral Drugs} & 425 & 350 \\
\text{Antifungal Agents} & 260 & 159 \\
\text{Antidepressants} & 848 & 353 \\
\text{Antipsychotics} & 461 & 291 \\
\text{Antihistamines} & 840 & 302 \\
\text{Corticosteroids} & 999 & 50 \\
\text{Beta Blockers} & 392 & 250 \\
\text{Calcium Channel Blockers} & 874 & 178 \\
\text{ACE Inhibitors} & 695 & 313 \\
\text{Angiotensin II Receptor Blockers} & 405 & 378 \\
\text{Diuretics} & 320 & 94 \\
\text{Statins} & 913 & 97 \\
\text{Insulin} & 754 & 470 \\
\text{Anticoagulants} & 428 & 341 \\
\text{Antiepileptic Drugs} & 711 & 121 \\
\text{Antiemetics} & 998 & 61 \\
\end{array}
\]

Total capacity (from capacity.csv): $C = 4120$

The complete mathematical model is:

\[
\begin{align*}
\textbf{Maximize} \quad & \sum_{i \in I} v_i x_i \\
\textbf{subject to} \quad & \sum_{i \in I} w_i x_i \leq 4120 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
\]

Where:

- $v_i$ = Value of drug type $i$ (see table above)
- $w_i$ = Weight of drug type $i$ (see table above)
- $x_i$ = number of units of drug type $i$ to order daily (integer, $\geq 0$)
- $I$ = set of all 20 drug types listed above

All coefficients and identifiers are as retrieved from the supplied data.