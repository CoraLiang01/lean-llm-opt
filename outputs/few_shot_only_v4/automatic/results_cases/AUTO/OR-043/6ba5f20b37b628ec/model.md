Let $x_i$ be the number of units of drug $i$ to order each day, where $i$ indexes the products listed below.

Objective:
\[
\max \sum_{i} v_i x_i
\]
where $v_i$ is the Value (benefit) of product $i$.

Subject to:

Stock capacity constraint:
\[
\sum_{i} w_i x_i \leq 520
\]
where $w_i$ is the Weight (stock space required) of product $i$.

Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:

\[
\begin{array}{lll}
\text{ProductName} & v_i\ (\text{Value}) & w_i\ (\text{Weight}) \\
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

Decision variables:
\[
x_i = \text{number of units of product } i \text{ to order each day},\quad x_i \in \mathbb{Z}_{\geq 0}
\]