Let $x_i$ be the number of units of drug $i$ to order each day, where $i$ indexes the products in the order given in products.csv.

Parameters (from products.csv):

\[
\begin{array}{lll}
\text{ProductName} & \text{Value } (v_i) & \text{Weight } (w_i) \\
\hline
\text{NSAIDs} & 250 & 913 \\
\text{Antirheumatic Drugs} & 178 & 754 \\
\text{Acetic Acid Derivatives} & 313 & 428 \\
\text{Antibiotics} & 301 & 711 \\
\text{Antiviral Drugs} & 425 & 350 \\
\text{Antifungal Agents} & 260 & 159 \\
\text{Antidepressants} & 848 & 353 \\
\text{Antipsychotics} & 934 & 291 \\
\text{Antihistamines} & 114 & 302 \\
\text{Corticosteroids} & 1357 & 50 \\
\text{Beta Blockers} & 156 & 250 \\
\text{Calcium Channel Blockers} & 1780 & 178 \\
\text{ACE Inhibitors} & 695 & 313 \\
\text{Angiotensin II Receptor Blockers} & 405 & 378 \\
\text{Diuretics} & 320 & 94 \\
\text{Statins} & 320 & 97 \\
\text{Insulin} & 1357 & 470 \\
\text{Anticoagulants} & 1357 & 341 \\
\text{Antiepileptic Drugs} & 405 & 121 \\
\text{Antiemetics} & 998 & 61 \\
\end{array}
\]

Total stock capacity (from capacity.csv):

\[
C = 520
\]

Model:

\[
\textbf{Decision variables:} \quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,20\}
\]

\[
\textbf{Objective:} \quad \max \sum_{i=1}^{20} v_i x_i
\]

\[
\textbf{Subject to:}
\]
\[
\sum_{i=1}^{20} w_i x_i \leq 520
\]
\[
x_i \geq 0 \text{ and integer}, \quad \forall i \in \{1,2,\ldots,20\}
\]

Where $v_i$ and $w_i$ are as listed above for each product in the order given.