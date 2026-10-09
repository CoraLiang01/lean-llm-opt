Let $I$ be the set of products, indexed in source order as follows:
\[
\begin{array}{lllr}
\text{Index} & \text{ProductName} & \text{Value} & \text{Weight} \\
1 & \text{NSAIDs} & 250 & 913 \\
2 & \text{Antirheumatic Drugs} & 178 & 754 \\
3 & \text{Acetic Acid Derivatives} & 313 & 428 \\
4 & \text{Antibiotics} & 301 & 711 \\
5 & \text{Antiviral Drugs} & 425 & 350 \\
6 & \text{Antifungal Agents} & 260 & 159 \\
7 & \text{Antidepressants} & 848 & 353 \\
8 & \text{Antipsychotics} & 934 & 291 \\
9 & \text{Antihistamines} & 114 & 302 \\
10 & \text{Corticosteroids} & 1357 & 50 \\
11 & \text{Beta Blockers} & 156 & 250 \\
12 & \text{Calcium Channel Blockers} & 1780 & 178 \\
13 & \text{ACE Inhibitors} & 695 & 313 \\
14 & \text{Angiotensin II Receptor Blockers} & 405 & 378 \\
15 & \text{Diuretics} & 320 & 94 \\
16 & \text{Statins} & 320 & 97 \\
17 & \text{Insulin} & 1357 & 470 \\
18 & \text{Anticoagulants} & 1357 & 341 \\
19 & \text{Antiepileptic Drugs} & 405 & 121 \\
20 & \text{Antiemetics} & 998 & 61 \\
\end{array}
\]

Let $x_i \geq 0$ be the number of units of product $i$ to order each day (continuous).

Let $v_i$ be the Value (benefit) and $w_i$ the Weight (stock usage) for product $i$ as above.

Let $C = 520$ be the overall stock capacity.

Objective:
\[
\max \sum_{i=1}^{20} v_i x_i
\]

Subject to:
\[
\sum_{i=1}^{20} w_i x_i \leq 520
\]
\[
x_i \geq 0 \quad \forall i=1,\ldots,20
\]

Where:
\[
\begin{array}{lllr}
i & \text{ProductName} & v_i & w_i \\
1 & \text{NSAIDs} & 250 & 913 \\
2 & \text{Antirheumatic Drugs} & 178 & 754 \\
3 & \text{Acetic Acid Derivatives} & 313 & 428 \\
4 & \text{Antibiotics} & 301 & 711 \\
5 & \text{Antiviral Drugs} & 425 & 350 \\
6 & \text{Antifungal Agents} & 260 & 159 \\
7 & \text{Antidepressants} & 848 & 353 \\
8 & \text{Antipsychotics} & 934 & 291 \\
9 & \text{Antihistamines} & 114 & 302 \\
10 & \text{Corticosteroids} & 1357 & 50 \\
11 & \text{Beta Blockers} & 156 & 250 \\
12 & \text{Calcium Channel Blockers} & 1780 & 178 \\
13 & \text{ACE Inhibitors} & 695 & 313 \\
14 & \text{Angiotensin II Receptor Blockers} & 405 & 378 \\
15 & \text{Diuretics} & 320 & 94 \\
16 & \text{Statins} & 320 & 97 \\
17 & \text{Insulin} & 1357 & 470 \\
18 & \text{Anticoagulants} & 1357 & 341 \\
19 & \text{Antiepileptic Drugs} & 405 & 121 \\
20 & \text{Antiemetics} & 998 & 61 \\
\end{array}
\]

All coefficients and identifiers are preserved in source order.