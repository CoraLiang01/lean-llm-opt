Let $x_i$ be the number of units of drug type $i$ to order daily, where $i$ indexes the following drug types in the order given:

1. NSAIDs
2. Antirheumatic Drugs
3. Acetic Acid Derivatives
4. Antibiotics
5. Antiviral Drugs
6. Antifungal Agents
7. Antidepressants
8. Antipsychotics
9. Antihistamines
10. Corticosteroids
11. Beta Blockers
12. Calcium Channel Blockers
13. ACE Inhibitors
14. Angiotensin II Receptor Blockers
15. Diuretics
16. Statins
17. Insulin
18. Anticoagulants
19. Antiepileptic Drugs
20. Antiemetics

The benefit coefficients and weights are as follows:

| Drug Type                          | Value | Weight |
|-------------------------------------|-------|--------|
| NSAIDs                             | 585   | 50     |
| Antirheumatic Drugs                | 557   | 329    |
| Acetic Acid Derivatives            | 963   | 410    |
| Antibiotics                        | 301   | 452    |
| Antiviral Drugs                    | 425   | 350    |
| Antifungal Agents                  | 260   | 159    |
| Antidepressants                    | 848   | 353    |
| Antipsychotics                     | 461   | 291    |
| Antihistamines                     | 840   | 302    |
| Corticosteroids                    | 999   | 50     |
| Beta Blockers                      | 392   | 250    |
| Calcium Channel Blockers           | 874   | 178    |
| ACE Inhibitors                     | 695   | 313    |
| Angiotensin II Receptor Blockers   | 405   | 378    |
| Diuretics                          | 320   | 94     |
| Statins                            | 913   | 97     |
| Insulin                            | 754   | 470    |
| Anticoagulants                     | 428   | 341    |
| Antiepileptic Drugs                | 711   | 121    |
| Antiemetics                        | 998   | 61     |

The total inventory capacity is $4120$ (from capacity.csv).

The complete mathematical model is:

$$
\begin{align*}
\max\quad & 585\,x_1 + 557\,x_2 + 963\,x_3 + 301\,x_4 + 425\,x_5 + 260\,x_6 + 848\,x_7 + 461\,x_8 \\
& + 840\,x_9 + 999\,x_{10} + 392\,x_{11} + 874\,x_{12} + 695\,x_{13} + 405\,x_{14} \\
& + 320\,x_{15} + 913\,x_{16} + 754\,x_{17} + 428\,x_{18} + 711\,x_{19} + 998\,x_{20} \\
\text{s.t.}\quad & 50\,x_1 + 329\,x_2 + 410\,x_3 + 452\,x_4 + 350\,x_5 + 159\,x_6 + 353\,x_7 + 291\,x_8 \\
& + 302\,x_9 + 50\,x_{10} + 250\,x_{11} + 178\,x_{12} + 313\,x_{13} + 378\,x_{14} \\
& + 94\,x_{15} + 97\,x_{16} + 470\,x_{17} + 341\,x_{18} + 121\,x_{19} + 61\,x_{20} \leq 4120 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad i=1,\ldots,20
\end{align*}
$$

Where:
- $x_i$ = number of units of drug type $i$ to order daily (integer, $\geq 0$)
- The objective maximizes total benefit.
- The constraint ensures the total weight of all ordered drugs does not exceed $4120$.