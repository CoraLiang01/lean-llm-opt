Let $x_i$ be the number of units of drug $i$ to order each day, where $i$ indexes the products in the order given below.

#### Objective Function

\[
\max \Big(
250\,x_{\text{NSAIDs}}
+ 178\,x_{\text{Antirheumatic Drugs}}
+ 313\,x_{\text{Acetic Acid Derivatives}}
+ 301\,x_{\text{Antibiotics}}
+ 425\,x_{\text{Antiviral Drugs}}
+ 260\,x_{\text{Antifungal Agents}}
+ 848\,x_{\text{Antidepressants}}
+ 934\,x_{\text{Antipsychotics}}
+ 114\,x_{\text{Antihistamines}}
+ 1357\,x_{\text{Corticosteroids}}
+ 156\,x_{\text{Beta Blockers}}
+ 1780\,x_{\text{Calcium Channel Blockers}}
+ 695\,x_{\text{ACE Inhibitors}}
+ 405\,x_{\text{Angiotensin II Receptor Blockers}}
+ 320\,x_{\text{Diuretics}}
+ 320\,x_{\text{Statins}}
+ 1357\,x_{\text{Insulin}}
+ 1357\,x_{\text{Anticoagulants}}
+ 405\,x_{\text{Antiepileptic Drugs}}
+ 998\,x_{\text{Antiemetics}}
\Big)
\]

#### Constraint

\[
913\,x_{\text{NSAIDs}}
+ 754\,x_{\text{Antirheumatic Drugs}}
+ 428\,x_{\text{Acetic Acid Derivatives}}
+ 711\,x_{\text{Antibiotics}}
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
\leq 520
\]

#### Variable Domains

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

#### Product Indexing (in source order)

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