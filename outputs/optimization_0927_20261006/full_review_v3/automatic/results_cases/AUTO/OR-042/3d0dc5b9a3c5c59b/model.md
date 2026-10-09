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

Let $v_i$ be the benefit coefficient and $w_i$ the weight per unit for drug $i$ as given below.

##### Objective Function

\[
\max \Big(
585x_1 + 557x_2 + 963x_3 + 301x_4 + 425x_5 + 260x_6 + 848x_7 + 461x_8 + 840x_9 + 999x_{10} + 392x_{11} + 874x_{12} + 695x_{13} + 405x_{14} + 320x_{15} + 913x_{16} + 754x_{17} + 428x_{18} + 711x_{19} + 998x_{20}
\Big)
\]

##### Subject to

\[
50x_1 + 329x_2 + 410x_3 + 452x_4 + 350x_5 + 159x_6 + 353x_7 + 291x_8 + 302x_9 + 50x_{10} + 250x_{11} + 178x_{12} + 313x_{13} + 378x_{14} + 94x_{15} + 97x_{16} + 470x_{17} + 341x_{18} + 121x_{19} + 61x_{20} \leq 4120
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
\]

##### Drug Data

| $i$ | ProductName                           | $v_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------------------------|---------------|---------------|
| 1   | NSAIDs                               | 585           | 50            |
| 2   | Antirheumatic Drugs                  | 557           | 329           |
| 3   | Acetic Acid Derivatives              | 963           | 410           |
| 4   | Antibiotics                          | 301           | 452           |
| 5   | Antiviral Drugs                      | 425           | 350           |
| 6   | Antifungal Agents                    | 260           | 159           |
| 7   | Antidepressants                      | 848           | 353           |
| 8   | Antipsychotics                       | 461           | 291           |
| 9   | Antihistamines                       | 840           | 302           |
| 10  | Corticosteroids                      | 999           | 50            |
| 11  | Beta Blockers                        | 392           | 250           |
| 12  | Calcium Channel Blockers             | 874           | 178           |
| 13  | ACE Inhibitors                       | 695           | 313           |
| 14  | Angiotensin II Receptor Blockers     | 405           | 378           |
| 15  | Diuretics                            | 320           | 94            |
| 16  | Statins                              | 913           | 97            |
| 17  | Insulin                              | 754           | 470           |
| 18  | Anticoagulants                       | 428           | 341           |
| 19  | Antiepileptic Drugs                  | 711           | 121           |
| 20  | Antiemetics                          | 998           | 61            |