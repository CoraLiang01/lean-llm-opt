##### Decision Variables

Let $x_i \in \mathbb{Z}_{\geq 0}$ denote the number of units of drug type $i$ to order daily, for each $i$ in the set of drug types $I$.

##### Parameters

Let $I$ be the set of drug types:
- NSAIDs
- Antirheumatic Drugs
- Acetic Acid Derivatives
- Antibiotics
- Antiviral Drugs
- Antifungal Agents
- Antidepressants
- Antipsychotics
- Antihistamines
- Corticosteroids
- Beta Blockers
- Calcium Channel Blockers
- ACE Inhibitors
- Angiotensin II Receptor Blockers
- Diuretics
- Statins
- Insulin
- Anticoagulants
- Antiepileptic Drugs
- Antiemetics

Let $v_i$ be the benefit coefficient of drug type $i$:

\[
\begin{align*}
v_{\text{NSAIDs}} &= 585 \\
v_{\text{Antirheumatic Drugs}} &= 557 \\
v_{\text{Acetic Acid Derivatives}} &= 963 \\
v_{\text{Antibiotics}} &= 301 \\
v_{\text{Antiviral Drugs}} &= 425 \\
v_{\text{Antifungal Agents}} &= 260 \\
v_{\text{Antidepressants}} &= 848 \\
v_{\text{Antipsychotics}} &= 461 \\
v_{\text{Antihistamines}} &= 840 \\
v_{\text{Corticosteroids}} &= 999 \\
v_{\text{Beta Blockers}} &= 392 \\
v_{\text{Calcium Channel Blockers}} &= 874 \\
v_{\text{ACE Inhibitors}} &= 695 \\
v_{\text{Angiotensin II Receptor Blockers}} &= 405 \\
v_{\text{Diuretics}} &= 320 \\
v_{\text{Statins}} &= 913 \\
v_{\text{Insulin}} &= 754 \\
v_{\text{Anticoagulants}} &= 428 \\
v_{\text{Antiepileptic Drugs}} &= 711 \\
v_{\text{Antiemetics}} &= 998 \\
\end{align*}
\]

Let $w_i$ be the weight per unit of drug type $i$:

\[
\begin{align*}
w_{\text{NSAIDs}} &= 50 \\
w_{\text{Antirheumatic Drugs}} &= 329 \\
w_{\text{Acetic Acid Derivatives}} &= 410 \\
w_{\text{Antibiotics}} &= 452 \\
w_{\text{Antiviral Drugs}} &= 350 \\
w_{\text{Antifungal Agents}} &= 159 \\
w_{\text{Antidepressants}} &= 353 \\
w_{\text{Antipsychotics}} &= 291 \\
w_{\text{Antihistamines}} &= 302 \\
w_{\text{Corticosteroids}} &= 50 \\
w_{\text{Beta Blockers}} &= 250 \\
w_{\text{Calcium Channel Blockers}} &= 178 \\
w_{\text{ACE Inhibitors}} &= 313 \\
w_{\text{Angiotensin II Receptor Blockers}} &= 378 \\
w_{\text{Diuretics}} &= 94 \\
w_{\text{Statins}} &= 97 \\
w_{\text{Insulin}} &= 470 \\
w_{\text{Anticoagulants}} &= 341 \\
w_{\text{Antiepileptic Drugs}} &= 121 \\
w_{\text{Antiemetics}} &= 61 \\
\end{align*}
\]

Let $C = 4120$ be the total inventory capacity.

##### Objective Function

\[
\max \sum_{i \in I} v_i x_i
\]

##### Constraints

1. Inventory capacity constraint:
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. Integer and nonnegativity constraints:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & 585\,x_{\text{NSAIDs}} + 557\,x_{\text{Antirheumatic Drugs}} + 963\,x_{\text{Acetic Acid Derivatives}} + 301\,x_{\text{Antibiotics}} + 425\,x_{\text{Antiviral Drugs}} \\
& + 260\,x_{\text{Antifungal Agents}} + 848\,x_{\text{Antidepressants}} + 461\,x_{\text{Antipsychotics}} + 840\,x_{\text{Antihistamines}} + 999\,x_{\text{Corticosteroids}} \\
& + 392\,x_{\text{Beta Blockers}} + 874\,x_{\text{Calcium Channel Blockers}} + 695\,x_{\text{ACE Inhibitors}} + 405\,x_{\text{Angiotensin II Receptor Blockers}} \\
& + 320\,x_{\text{Diuretics}} + 913\,x_{\text{Statins}} + 754\,x_{\text{Insulin}} + 428\,x_{\text{Anticoagulants}} + 711\,x_{\text{Antiepileptic Drugs}} + 998\,x_{\text{Antiemetics}} \\
\text{s.t.} \quad & 50\,x_{\text{NSAIDs}} + 329\,x_{\text{Antirheumatic Drugs}} + 410\,x_{\text{Acetic Acid Derivatives}} + 452\,x_{\text{Antibiotics}} + 350\,x_{\text{Antiviral Drugs}} \\
& + 159\,x_{\text{Antifungal Agents}} + 353\,x_{\text{Antidepressants}} + 291\,x_{\text{Antipsychotics}} + 302\,x_{\text{Antihistamines}} + 50\,x_{\text{Corticosteroids}} \\
& + 250\,x_{\text{Beta Blockers}} + 178\,x_{\text{Calcium Channel Blockers}} + 313\,x_{\text{ACE Inhibitors}} + 378\,x_{\text{Angiotensin II Receptor Blockers}} \\
& + 94\,x_{\text{Diuretics}} + 97\,x_{\text{Statins}} + 470\,x_{\text{Insulin}} + 341\,x_{\text{Anticoagulants}} + 121\,x_{\text{Antiepileptic Drugs}} + 61\,x_{\text{Antiemetics}} \leq 4120 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]