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

| Drug Type                              | Value | Weight |
|----------------------------------------|-------|--------|
| NSAIDs                                 | 585   | 50     |
| Antirheumatic Drugs                    | 557   | 329    |
| Acetic Acid Derivatives                | 963   | 410    |
| Antibiotics                            | 301   | 452    |
| Antiviral Drugs                        | 425   | 350    |
| Antifungal Agents                      | 260   | 159    |
| Antidepressants                        | 848   | 353    |
| Antipsychotics                         | 461   | 291    |
| Antihistamines                         | 840   | 302    |
| Corticosteroids                        | 999   | 50     |
| Beta Blockers                          | 392   | 250    |
| Calcium Channel Blockers               | 874   | 178    |
| ACE Inhibitors                         | 695   | 313    |
| Angiotensin II Receptor Blockers       | 405   | 378    |
| Diuretics                              | 320   | 94     |
| Statins                                | 913   | 97     |
| Insulin                                | 754   | 470    |
| Anticoagulants                         | 428   | 341    |
| Antiepileptic Drugs                    | 711   | 121    |
| Antiemetics                            | 998   | 61     |

The total inventory capacity is $4120$ (from capacity.csv).

The mathematical model is:

$$
\begin{align*}
\max\quad & 585x_1 + 557x_2 + 963x_3 + 301x_4 + 425x_5 + 260x_6 + 848x_7 + 461x_8 + 840x_9 + 999x_{10} \\
& + 392x_{11} + 874x_{12} + 695x_{13} + 405x_{14} + 320x_{15} + 913x_{16} + 754x_{17} + 428x_{18} + 711x_{19} + 998x_{20} \\
\text{s.t.}\quad & 50x_1 + 329x_2 + 410x_3 + 452x_4 + 350x_5 + 159x_6 + 353x_7 + 291x_8 + 302x_9 + 50x_{10} \\
& + 250x_{11} + 178x_{12} + 313x_{13} + 378x_{14} + 94x_{15} + 97x_{16} + 470x_{17} + 341x_{18} + 121x_{19} + 61x_{20} \leq 4120 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i=1,\ldots,20
\end{align*}
$$

Where:
- $x_i$ = number of units of drug type $i$ to order daily (integer, $\geq 0$)
- The objective maximizes total benefit.
- The constraint ensures the total weight of all ordered drugs does not exceed the overall capacity of $4120$.