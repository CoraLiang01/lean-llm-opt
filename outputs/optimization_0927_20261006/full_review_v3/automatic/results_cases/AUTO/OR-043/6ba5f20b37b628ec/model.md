Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given below.

**Objective:**
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

**Subject to:**

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

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Where:**

- $x_{\text{NSAIDs}}$: Number of units of NSAIDs to order each day
- $x_{\text{Antirheumatic Drugs}}$: Number of units of Antirheumatic Drugs to order each day
- $x_{\text{Acetic Acid Derivatives}}$: Number of units of Acetic Acid Derivatives to order each day
- $x_{\text{Antibiotics}}$: Number of units of Antibiotics to order each day
- $x_{\text{Antiviral Drugs}}$: Number of units of Antiviral Drugs to order each day
- $x_{\text{Antifungal Agents}}$: Number of units of Antifungal Agents to order each day
- $x_{\text{Antidepressants}}$: Number of units of Antidepressants to order each day
- $x_{\text{Antipsychotics}}$: Number of units of Antipsychotics to order each day
- $x_{\text{Antihistamines}}$: Number of units of Antihistamines to order each day
- $x_{\text{Corticosteroids}}$: Number of units of Corticosteroids to order each day
- $x_{\text{Beta Blockers}}$: Number of units of Beta Blockers to order each day
- $x_{\text{Calcium Channel Blockers}}$: Number of units of Calcium Channel Blockers to order each day
- $x_{\text{ACE Inhibitors}}$: Number of units of ACE Inhibitors to order each day
- $x_{\text{Angiotensin II Receptor Blockers}}$: Number of units of Angiotensin II Receptor Blockers to order each day
- $x_{\text{Diuretics}}$: Number of units of Diuretics to order each day
- $x_{\text{Statins}}$: Number of units of Statins to order each day
- $x_{\text{Insulin}}$: Number of units of Insulin to order each day
- $x_{\text{Anticoagulants}}$: Number of units of Anticoagulants to order each day
- $x_{\text{Antiepileptic Drugs}}$: Number of units of Antiepileptic Drugs to order each day
- $x_{\text{Antiemetics}}$: Number of units of Antiemetics to order each day

**Capacity:**
- Total stock capacity per day: $520$ units (as per the "Capacity" value in capacity.csv).

**Variable domains:**
- All $x_i$ are nonnegative integers.