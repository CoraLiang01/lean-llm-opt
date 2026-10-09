Let $x_i$ be the integer number of units of drug type $i$ to order daily, for each $i$ in the set of products below.

Maximize total benefit:
$$
\max \Big(
585\,x_{\text{NSAIDs}}
+ 557\,x_{\text{Antirheumatic Drugs}}
+ 963\,x_{\text{Acetic Acid Derivatives}}
+ 301\,x_{\text{Antibiotics}}
+ 425\,x_{\text{Antiviral Drugs}}
+ 260\,x_{\text{Antifungal Agents}}
+ 848\,x_{\text{Antidepressants}}
+ 461\,x_{\text{Antipsychotics}}
+ 840\,x_{\text{Antihistamines}}
+ 999\,x_{\text{Corticosteroids}}
+ 392\,x_{\text{Beta Blockers}}
+ 874\,x_{\text{Calcium Channel Blockers}}
+ 695\,x_{\text{ACE Inhibitors}}
+ 405\,x_{\text{Angiotensin II Receptor Blockers}}
+ 320\,x_{\text{Diuretics}}
+ 913\,x_{\text{Statins}}
+ 754\,x_{\text{Insulin}}
+ 428\,x_{\text{Anticoagulants}}
+ 711\,x_{\text{Antiepileptic Drugs}}
+ 998\,x_{\text{Antiemetics}}
\Big)
$$

Subject to the overall inventory capacity constraint:
$$
50\,x_{\text{NSAIDs}}
+ 329\,x_{\text{Antirheumatic Drugs}}
+ 410\,x_{\text{Acetic Acid Derivatives}}
+ 452\,x_{\text{Antibiotics}}
+ 350\,x_{\text{Antiviral Drugs}}
+ 159\,x_{\text{Antifungal Agents}}
+ 353\,x_{\text{Antidepressants}}
+ 291\,x_{\text{Antipsychotics}}
+ 302\,x_{\text{Antihistamines}}
+ 50\,x_{\text{Corticosteroids}}
+ 250\,x_{\text{Beta Blockers}}
+ 178\,x_{\text{Calcium Channel Blockers}}
+ 313\,x_{\text{ACE Inhibitors}}
+ 378\,x_{\text{Angiotensin II Receptor Blockers}}
+ 94\,x_{\text{Diuretics}}
+ 97\,x_{\text{Statins}}
+ 470\,x_{\text{Insulin}}
+ 341\,x_{\text{Anticoagulants}}
+ 121\,x_{\text{Antiepileptic Drugs}}
+ 61\,x_{\text{Antiemetics}}
\leq 4120
$$

Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all drug types } i
$$

Where the drug types and their coefficients (in source order) are:

| $i$ | Product Name | Value (Benefit) | Weight per unit |
|---|-------------------------------|------------------|-----------------|
| 1 | NSAIDs | 585 | 50 |
| 2 | Antirheumatic Drugs | 557 | 329 |
| 3 | Acetic Acid Derivatives | 963 | 410 |
| 4 | Antibiotics | 301 | 452 |
| 5 | Antiviral Drugs | 425 | 350 |
| 6 | Antifungal Agents | 260 | 159 |
| 7 | Antidepressants | 848 | 353 |
| 8 | Antipsychotics | 461 | 291 |
| 9 | Antihistamines | 840 | 302 |
| 10 | Corticosteroids | 999 | 50 |
| 11 | Beta Blockers | 392 | 250 |
| 12 | Calcium Channel Blockers | 874 | 178 |
| 13 | ACE Inhibitors | 695 | 313 |
| 14 | Angiotensin II Receptor Blockers | 405 | 378 |
| 15 | Diuretics | 320 | 94 |
| 16 | Statins | 913 | 97 |
| 17 | Insulin | 754 | 470 |
| 18 | Anticoagulants | 428 | 341 |
| 19 | Antiepileptic Drugs | 711 | 121 |
| 20 | Antiemetics | 998 | 61 |

All $x_i$ are integer and nonnegative. The total weight of all ordered units cannot exceed 4120. The objective is to maximize total benefit.