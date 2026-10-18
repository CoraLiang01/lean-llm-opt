Let $x_i$ be the number of units of drug $i$ to order each day, where $i$ indexes the products in the order given in the data.

Objective:
\[
\max \quad 250\,x_1 + 178\,x_2 + 313\,x_3 + 301\,x_4 + 425\,x_5 + 260\,x_6 + 848\,x_7 + 934\,x_8 + 114\,x_9 + 1357\,x_{10} + 156\,x_{11} + 1780\,x_{12} + 695\,x_{13} + 405\,x_{14} + 320\,x_{15} + 320\,x_{16} + 1357\,x_{17} + 1357\,x_{18} + 405\,x_{19} + 998\,x_{20}
\]

Subject to:

\[
913\,x_1 + 754\,x_2 + 428\,x_3 + 711\,x_4 + 350\,x_5 + 159\,x_6 + 353\,x_7 + 291\,x_8 + 302\,x_9 + 50\,x_{10} + 250\,x_{11} + 178\,x_{12} + 313\,x_{13} + 378\,x_{14} + 94\,x_{15} + 97\,x_{16} + 470\,x_{17} + 341\,x_{18} + 121\,x_{19} + 61\,x_{20} \leq 520
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
\]

Where the products are indexed as follows (in source order):

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

All variables $x_i$ are nonnegative integers.