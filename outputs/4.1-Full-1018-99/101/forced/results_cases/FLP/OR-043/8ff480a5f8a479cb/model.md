##### Decision Variables

Let $x_i \geq 0$ be the number of units of drug $i$ to order each day, for each product $i$ in the set $P$ of all products.

##### Parameters

Let $P$ be the set of products:

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

Let $v_i$ be the benefit (Value) per unit of product $i$:

\[
\begin{align*}
v_{\text{NSAIDs}} &= 250 \\
v_{\text{Antirheumatic Drugs}} &= 178 \\
v_{\text{Acetic Acid Derivatives}} &= 313 \\
v_{\text{Antibiotics}} &= 301 \\
v_{\text{Antiviral Drugs}} &= 425 \\
v_{\text{Antifungal Agents}} &= 260 \\
v_{\text{Antidepressants}} &= 848 \\
v_{\text{Antipsychotics}} &= 934 \\
v_{\text{Antihistamines}} &= 114 \\
v_{\text{Corticosteroids}} &= 1357 \\
v_{\text{Beta Blockers}} &= 156 \\
v_{\text{Calcium Channel Blockers}} &= 1780 \\
v_{\text{ACE Inhibitors}} &= 695 \\
v_{\text{Angiotensin II Receptor Blockers}} &= 405 \\
v_{\text{Diuretics}} &= 320 \\
v_{\text{Statins}} &= 320 \\
v_{\text{Insulin}} &= 1357 \\
v_{\text{Anticoagulants}} &= 1357 \\
v_{\text{Antiepileptic Drugs}} &= 405 \\
v_{\text{Antiemetics}} &= 998 \\
\end{align*}
\]

Let $w_i$ be the weight per unit of product $i$:

\[
\begin{align*}
w_{\text{NSAIDs}} &= 913 \\
w_{\text{Antirheumatic Drugs}} &= 754 \\
w_{\text{Acetic Acid Derivatives}} &= 428 \\
w_{\text{Antibiotics}} &= 711 \\
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

Let $C = 520$ be the overall stock capacity.

##### Objective Function

\[
\max \sum_{i \in P} v_i x_i
\]

##### Constraints

1. Overall stock capacity:
   \[
   \sum_{i \in P} w_i x_i \leq C
   \]
2. Nonnegativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in P} v_i x_i \\
\text{s.t.} \quad & \sum_{i \in P} w_i x_i \leq 520 \\
& x_i \geq 0 \quad \forall i \in P
\end{align*}
\]

Where the sets and parameters are as listed above.