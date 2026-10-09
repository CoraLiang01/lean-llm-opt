**Mathematical Model**

Let $i$ index the drug types, with $i$ corresponding to the following ProductName and parameters:

| $i$ | ProductName                              | Value ($v_i$) | Weight ($w_i$) |
|-----|------------------------------------------|---------------|---------------|
| 1   | NSAIDs                                  | 585           | 50            |
| 2   | Antirheumatic Drugs                     | 557           | 329           |
| 3   | Acetic Acid Derivatives                 | 963           | 410           |
| 4   | Antibiotics                             | 301           | 452           |
| 5   | Antiviral Drugs                         | 425           | 350           |
| 6   | Antifungal Agents                       | 260           | 159           |
| 7   | Antidepressants                         | 848           | 353           |
| 8   | Antipsychotics                          | 461           | 291           |
| 9   | Antihistamines                          | 840           | 302           |
| 10  | Corticosteroids                         | 999           | 50            |
| 11  | Beta Blockers                           | 392           | 250           |
| 12  | Calcium Channel Blockers                 | 874           | 178           |
| 13  | ACE Inhibitors                          | 695           | 313           |
| 14  | Angiotensin II Receptor Blockers        | 405           | 378           |
| 15  | Diuretics                               | 320           | 94            |
| 16  | Statins                                 | 913           | 97            |
| 17  | Insulin                                 | 754           | 470           |
| 18  | Anticoagulants                          | 428           | 341           |
| 19  | Antiepileptic Drugs                     | 711           | 121           |
| 20  | Antiemetics                             | 998           | 61            |

Let $x_i$ be the number of units of drug type $i$ to order daily. All $x_i$ are required to be nonnegative integers.

Let the total inventory capacity be $C = 4120$.

---

**Variables**

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
$$

---

**Objective**

Maximize total benefit:
$$
\max \sum_{i=1}^{20} v_i x_i
$$

That is,
\[
\max \Big(
585\,x_1 + 557\,x_2 + 963\,x_3 + 301\,x_4 + 425\,x_5 + 260\,x_6 + 848\,x_7 + 461\,x_8 + 840\,x_9 + 999\,x_{10} + 392\,x_{11} + 874\,x_{12} + 695\,x_{13} + 405\,x_{14} + 320\,x_{15} + 913\,x_{16} + 754\,x_{17} + 428\,x_{18} + 711\,x_{19} + 998\,x_{20}
\Big)
\]

---

**Constraint**

Total weight of all ordered units cannot exceed capacity:
$$
\sum_{i=1}^{20} w_i x_i \leq 4120
$$

That is,
\[
50\,x_1 + 329\,x_2 + 410\,x_3 + 452\,x_4 + 350\,x_5 + 159\,x_6 + 353\,x_7 + 291\,x_8 + 302\,x_9 + 50\,x_{10} + 250\,x_{11} + 178\,x_{12} + 313\,x_{13} + 378\,x_{14} + 94\,x_{15} + 97\,x_{16} + 470\,x_{17} + 341\,x_{18} + 121\,x_{19} + 61\,x_{20} \leq 4120
\]

---

**Variable Domains**

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
$$

---

**Complete Model**

\[
\begin{align*}
\max\quad & 585\,x_1 + 557\,x_2 + 963\,x_3 + 301\,x_4 + 425\,x_5 + 260\,x_6 + 848\,x_7 + 461\,x_8 + 840\,x_9 \\
& + 999\,x_{10} + 392\,x_{11} + 874\,x_{12} + 695\,x_{13} + 405\,x_{14} + 320\,x_{15} + 913\,x_{16} \\
& + 754\,x_{17} + 428\,x_{18} + 711\,x_{19} + 998\,x_{20} \\
\text{s.t.}\quad & 50\,x_1 + 329\,x_2 + 410\,x_3 + 452\,x_4 + 350\,x_5 + 159\,x_6 + 353\,x_7 + 291\,x_8 + 302\,x_9 \\
& + 50\,x_{10} + 250\,x_{11} + 178\,x_{12} + 313\,x_{13} + 378\,x_{14} + 94\,x_{15} + 97\,x_{16} \\
& + 470\,x_{17} + 341\,x_{18} + 121\,x_{19} + 61\,x_{20} \leq 4120 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
\end{align*}
\]