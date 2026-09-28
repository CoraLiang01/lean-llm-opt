Let $x_i$ be the number of units of drug $i$ to order each day, where $i$ indexes the products in the order given below. All $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
$$
\max \; 250x_{\text{NSAIDs}} + 178x_{\text{Antirheumatic Drugs}} + 313x_{\text{Acetic Acid Derivatives}} + 301x_{\text{Antibiotics}} + 425x_{\text{Antiviral Drugs}} + 260x_{\text{Antifungal Agents}} + 848x_{\text{Antidepressants}} + 934x_{\text{Antipsychotics}} + 114x_{\text{Antihistamines}} + 1357x_{\text{Corticosteroids}} + 156x_{\text{Beta Blockers}} + 1780x_{\text{Calcium Channel Blockers}} + 695x_{\text{ACE Inhibitors}} + 405x_{\text{Angiotensin II Receptor Blockers}} + 320x_{\text{Diuretics}} + 320x_{\text{Statins}} + 1357x_{\text{Insulin}} + 1357x_{\text{Anticoagulants}} + 405x_{\text{Antiepileptic Drugs}} + 998x_{\text{Antiemetics}}
$$

Subject to:

$$
913x_{\text{NSAIDs}} + 754x_{\text{Antirheumatic Drugs}} + 428x_{\text{Acetic Acid Derivatives}} + 711x_{\text{Antibiotics}} + 350x_{\text{Antiviral Drugs}} + 159x_{\text{Antifungal Agents}} + 353x_{\text{Antidepressants}} + 291x_{\text{Antipsychotics}} + 302x_{\text{Antihistamines}} + 50x_{\text{Corticosteroids}} + 250x_{\text{Beta Blockers}} + 178x_{\text{Calcium Channel Blockers}} + 313x_{\text{ACE Inhibitors}} + 378x_{\text{Angiotensin II Receptor Blockers}} + 94x_{\text{Diuretics}} + 97x_{\text{Statins}} + 470x_{\text{Insulin}} + 341x_{\text{Anticoagulants}} + 121x_{\text{Antiepileptic Drugs}} + 61x_{\text{Antiemetics}} \leq 520
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where the products $i$ are, in order:
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

All coefficients and identifiers are as retrieved from the data.