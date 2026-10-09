Let $x_i$ be the number of units of drug type $i$ to order daily, where $i$ indexes the following drug types in the order given below.

**Objective:**
\[
\max \Big(
585\, x_{\text{NSAIDs}}
+ 557\, x_{\text{Antirheumatic Drugs}}
+ 963\, x_{\text{Acetic Acid Derivatives}}
+ 301\, x_{\text{Antibiotics}}
+ 425\, x_{\text{Antiviral Drugs}}
+ 260\, x_{\text{Antifungal Agents}}
+ 848\, x_{\text{Antidepressants}}
+ 461\, x_{\text{Antipsychotics}}
+ 840\, x_{\text{Antihistamines}}
+ 999\, x_{\text{Corticosteroids}}
+ 392\, x_{\text{Beta Blockers}}
+ 874\, x_{\text{Calcium Channel Blockers}}
+ 695\, x_{\text{ACE Inhibitors}}
+ 405\, x_{\text{Angiotensin II Receptor Blockers}}
+ 320\, x_{\text{Diuretics}}
+ 913\, x_{\text{Statins}}
+ 754\, x_{\text{Insulin}}
+ 428\, x_{\text{Anticoagulants}}
+ 711\, x_{\text{Antiepileptic Drugs}}
+ 998\, x_{\text{Antiemetics}}
\Big)
\]

**Subject to:**
\[
50\, x_{\text{NSAIDs}}
+ 329\, x_{\text{Antirheumatic Drugs}}
+ 410\, x_{\text{Acetic Acid Derivatives}}
+ 452\, x_{\text{Antibiotics}}
+ 350\, x_{\text{Antiviral Drugs}}
+ 159\, x_{\text{Antifungal Agents}}
+ 353\, x_{\text{Antidepressants}}
+ 291\, x_{\text{Antipsychotics}}
+ 302\, x_{\text{Antihistamines}}
+ 50\, x_{\text{Corticosteroids}}
+ 250\, x_{\text{Beta Blockers}}
+ 178\, x_{\text{Calcium Channel Blockers}}
+ 313\, x_{\text{ACE Inhibitors}}
+ 378\, x_{\text{Angiotensin II Receptor Blockers}}
+ 94\, x_{\text{Diuretics}}
+ 97\, x_{\text{Statins}}
+ 470\, x_{\text{Insulin}}
+ 341\, x_{\text{Anticoagulants}}
+ 121\, x_{\text{Antiepileptic Drugs}}
+ 61\, x_{\text{Antiemetics}}
\leq 4120
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

---

**Parameter Table (in source order):**

| ProductName                        | Value | Weight |
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

**Capacity:**

- Total allowable weight: $4120$ units

**Decision variables:**

- $x_i$ = number of units of drug type $i$ to order daily, integer, $x_i \geq 0$ for all $i$ (as listed above).