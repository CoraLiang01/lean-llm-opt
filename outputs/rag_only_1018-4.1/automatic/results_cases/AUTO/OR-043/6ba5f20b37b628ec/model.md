Let the set of products (in original file order) be indexed by i, with the following data for each product i:

| i  | ProductName                          | Value (benefit per unit) | Weight (stock units per unit) |
|----|--------------------------------------|--------------------------|-------------------------------|
| 1  | NSAIDs                              | 250                      | 913                           |
| 2  | Antirheumatic Drugs                  | 178                      | 754                           |
| 3  | Acetic Acid Derivatives              | 313                      | 428                           |
| 4  | Antibiotics                          | 301                      | 711                           |
| 5  | Antiviral Drugs                      | 425                      | 350                           |
| 6  | Antifungal Agents                    | 260                      | 159                           |
| 7  | Antidepressants                      | 848                      | 353                           |
| 8  | Antipsychotics                       | 934                      | 291                           |
| 9  | Antihistamines                       | 114                      | 302                           |
| 10 | Corticosteroids                      | 1357                     | 50                            |
| 11 | Beta Blockers                        | 156                      | 250                           |
| 12 | Calcium Channel Blockers              | 1780                     | 178                           |
| 13 | ACE Inhibitors                       | 695                      | 313                           |
| 14 | Angiotensin II Receptor Blockers     | 405                      | 378                           |
| 15 | Diuretics                            | 320                      | 94                            |
| 16 | Statins                              | 320                      | 97                            |
| 17 | Insulin                              | 1357                     | 470                           |
| 18 | Anticoagulants                       | 1357                     | 341                           |
| 19 | Antiepileptic Drugs                  | 405                      | 121                           |
| 20 | Antiemetics                          | 998                      | 61                            |

Let x_i = number of units of product i to order each day (decision variable, integer, x_i ≥ 0 for all i).

Let the total stock capacity be C = 520 (from capacity.csv).

The mathematical optimization model is:

Variables:
- For i = 1,...,20: x_i ∈ {0, 1, 2, ...}

Objective:
Maximize total benefit:
\[
\text{Maximize} \quad 250x_1 + 178x_2 + 313x_3 + 301x_4 + 425x_5 + 260x_6 + 848x_7 + 934x_8 + 114x_9 + 1357x_{10} + 156x_{11} + 1780x_{12} + 695x_{13} + 405x_{14} + 320x_{15} + 320x_{16} + 1357x_{17} + 1357x_{18} + 405x_{19} + 998x_{20}
\]

Subject to the overall stock capacity constraint:
\[
913x_1 + 754x_2 + 428x_3 + 711x_4 + 350x_5 + 159x_6 + 353x_7 + 291x_8 + 302x_9 + 50x_{10} + 250x_{11} + 178x_{12} + 313x_{13} + 378x_{14} + 94x_{15} + 97x_{16} + 470x_{17} + 341x_{18} + 121x_{19} + 61x_{20} \leq 520
\]

and
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all } i = 1, ..., 20
\]

where each x_i corresponds to the product in the original file order as listed above.