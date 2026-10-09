Let $x_i$ be the number of units of product $i$ to order each day, for each product $i$ in the table below.

**Objective:**
\[
\max \quad 250x_{\text{NSAIDs}} + 178x_{\text{Antirheumatic Drugs}} + 313x_{\text{Acetic Acid Derivatives}} + 301x_{\text{Antibiotics}} + 425x_{\text{Antiviral Drugs}} + 260x_{\text{Antifungal Agents}} + 848x_{\text{Antidepressants}} + 934x_{\text{Antipsychotics}} + 114x_{\text{Antihistamines}} + 1357x_{\text{Corticosteroids}} + 156x_{\text{Beta Blockers}} + 1780x_{\text{Calcium Channel Blockers}} + 695x_{\text{ACE Inhibitors}} + 405x_{\text{Angiotensin II Receptor Blockers}} + 320x_{\text{Diuretics}} + 320x_{\text{Statins}} + 1357x_{\text{Insulin}} + 1357x_{\text{Anticoagulants}} + 405x_{\text{Antiepileptic Drugs}} + 998x_{\text{Antiemetics}}
\]

**Subject to:**

**Capacity constraint:**
\[
913x_{\text{NSAIDs}} + 754x_{\text{Antirheumatic Drugs}} + 428x_{\text{Acetic Acid Derivatives}} + 711x_{\text{Antibiotics}} + 350x_{\text{Antiviral Drugs}} + 159x_{\text{Antifungal Agents}} + 353x_{\text{Antidepressants}} + 291x_{\text{Antipsychotics}} + 302x_{\text{Antihistamines}} + 50x_{\text{Corticosteroids}} + 250x_{\text{Beta Blockers}} + 178x_{\text{Calcium Channel Blockers}} + 313x_{\text{ACE Inhibitors}} + 378x_{\text{Angiotensin II Receptor Blockers}} + 94x_{\text{Diuretics}} + 97x_{\text{Statins}} + 470x_{\text{Insulin}} + 341x_{\text{Anticoagulants}} + 121x_{\text{Antiepileptic Drugs}} + 61x_{\text{Antiemetics}} \leq 520
\]

**Nonnegativity and integrality:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

**Where:**

| ProductName                        | Value | Weight |
|-------------------------------------|-------|--------|
| NSAIDs                             | 250   | 913    |
| Antirheumatic Drugs                | 178   | 754    |
| Acetic Acid Derivatives            | 313   | 428    |
| Antibiotics                        | 301   | 711    |
| Antiviral Drugs                    | 425   | 350    |
| Antifungal Agents                  | 260   | 159    |
| Antidepressants                    | 848   | 353    |
| Antipsychotics                     | 934   | 291    |
| Antihistamines                     | 114   | 302    |
| Corticosteroids                    | 1357  | 50     |
| Beta Blockers                      | 156   | 250    |
| Calcium Channel Blockers           | 1780  | 178    |
| ACE Inhibitors                     | 695   | 313    |
| Angiotensin II Receptor Blockers   | 405   | 378    |
| Diuretics                          | 320   | 94     |
| Statins                            | 320   | 97     |
| Insulin                            | 1357  | 470    |
| Anticoagulants                     | 1357  | 341    |
| Antiepileptic Drugs                | 405   | 121    |
| Antiemetics                        | 998   | 61     |