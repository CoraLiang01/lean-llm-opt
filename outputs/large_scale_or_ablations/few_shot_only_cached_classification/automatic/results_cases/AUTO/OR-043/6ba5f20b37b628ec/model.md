##### Sets and Indices

Let $i$ index the drug products as listed in the table below.

##### Parameters

For each drug $i$:

- $v_i$ = Value (benefit) of one unit of drug $i$
- $w_i$ = Weight (stock space required) for one unit of drug $i$

Let $C$ = 520 (overall stock capacity)

##### Decision Variables

For each drug $i$:

- $x_i$ = number of units of drug $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Data

| $i$ | ProductName                        | $v_i$ | $w_i$ |
|-----|------------------------------------|-------|-------|
| 1   | NSAIDs                             | 250   | 913   |
| 2   | Antirheumatic Drugs                | 178   | 754   |
| 3   | Acetic Acid Derivatives            | 313   | 428   |
| 4   | Antibiotics                        | 301   | 711   |
| 5   | Antiviral Drugs                    | 425   | 350   |
| 6   | Antifungal Agents                  | 260   | 159   |
| 7   | Antidepressants                    | 848   | 353   |
| 8   | Antipsychotics                     | 934   | 291   |
| 9   | Antihistamines                     | 114   | 302   |
| 10  | Corticosteroids                    | 1357  | 50    |
| 11  | Beta Blockers                      | 156   | 250   |
| 12  | Calcium Channel Blockers           | 1780  | 178   |
| 13  | ACE Inhibitors                     | 695   | 313   |
| 14  | Angiotensin II Receptor Blockers   | 405   | 378   |
| 15  | Diuretics                          | 320   | 94    |
| 16  | Statins                            | 320   | 97    |
| 17  | Insulin                            | 1357  | 470   |
| 18  | Anticoagulants                     | 1357  | 341   |
| 19  | Antiepileptic Drugs                | 405   | 121   |
| 20  | Antiemetics                        | 998   | 61    |

##### Mathematical Model

Objective:
$$
\max \sum_{i=1}^{20} v_i x_i
$$

Subject to:
$$
\sum_{i=1}^{20} w_i x_i \leq 520
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
$$

Where:

- $v_i$ and $w_i$ are as given in the table above for each product.
- $C = 520$ is the total stock capacity.