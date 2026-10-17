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

Let $v_i$ be the benefit coefficient for drug type $i$:

\[
\begin{aligned}
&v_{\text{NSAIDs}} = 585 \\
&v_{\text{Antirheumatic Drugs}} = 557 \\
&v_{\text{Acetic Acid Derivatives}} = 963 \\
&v_{\text{Antibiotics}} = 301 \\
&v_{\text{Antiviral Drugs}} = 425 \\
&v_{\text{Antifungal Agents}} = 260 \\
&v_{\text{Antidepressants}} = 848 \\
&v_{\text{Antipsychotics}} = 461 \\
&v_{\text{Antihistamines}} = 840 \\
&v_{\text{Corticosteroids}} = 999 \\
&v_{\text{Beta Blockers}} = 392 \\
&v_{\text{Calcium Channel Blockers}} = 874 \\
&v_{\text{ACE Inhibitors}} = 695 \\
&v_{\text{Angiotensin II Receptor Blockers}} = 405 \\
&v_{\text{Diuretics}} = 320 \\
&v_{\text{Statins}} = 913 \\
&v_{\text{Insulin}} = 754 \\
&v_{\text{Anticoagulants}} = 428 \\
&v_{\text{Antiepileptic Drugs}} = 711 \\
&v_{\text{Antiemetics}} = 998 \\
\end{aligned}
\]

Let $w_i$ be the per-unit weight for drug type $i$:

\[
\begin{aligned}
&w_{\text{NSAIDs}} = 50 \\
&w_{\text{Antirheumatic Drugs}} = 329 \\
&w_{\text{Acetic Acid Derivatives}} = 410 \\
&w_{\text{Antibiotics}} = 452 \\
&w_{\text{Antiviral Drugs}} = 350 \\
&w_{\text{Antifungal Agents}} = 159 \\
&w_{\text{Antidepressants}} = 353 \\
&w_{\text{Antipsychotics}} = 291 \\
&w_{\text{Antihistamines}} = 302 \\
&w_{\text{Corticosteroids}} = 50 \\
&w_{\text{Beta Blockers}} = 250 \\
&w_{\text{Calcium Channel Blockers}} = 178 \\
&w_{\text{ACE Inhibitors}} = 313 \\
&w_{\text{Angiotensin II Receptor Blockers}} = 378 \\
&w_{\text{Diuretics}} = 94 \\
&w_{\text{Statins}} = 97 \\
&w_{\text{Insulin}} = 470 \\
&w_{\text{Anticoagulants}} = 341 \\
&w_{\text{Antiepileptic Drugs}} = 121 \\
&w_{\text{Antiemetics}} = 61 \\
\end{aligned}
\]

Let $C = 4120$ be the total inventory capacity.

##### Objective Function

\[
\max \sum_{i \in I} v_i x_i
\]

##### Constraints

1. Inventory capacity constraint:
   \[
   \sum_{i \in I} w_i x_i \leq 4120
   \]

2. Integer and nonnegativity constraints:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i \in I} v_i x_i \\
\text{s.t.}\quad & \sum_{i \in I} w_i x_i \leq 4120 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\end{align*}
\]

where $I$ and all parameter values are as listed above.