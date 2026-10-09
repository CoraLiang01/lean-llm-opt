**Sets and Indices:**
- Let $i$ index the 20 drug products as listed in products.csv.

**Parameters:**
- $v_i$ = Value of product $i$ (from products.csv)
- $w_i$ = Weight of product $i$ (from products.csv)
- $C$ = 520 (overall stock capacity from capacity.csv)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day

**Objective:**
\[
\max \sum_{i} v_i x_i
\]
where the $v_i$ are as follows (in file order):

\[
\begin{align*}
&v_{\text{NSAIDs}} = 250 \\
&v_{\text{Antirheumatic Drugs}} = 178 \\
&v_{\text{Acetic Acid Derivatives}} = 313 \\
&v_{\text{Antibiotics}} = 301 \\
&v_{\text{Antiviral Drugs}} = 425 \\
&v_{\text{Antifungal Agents}} = 260 \\
&v_{\text{Antidepressants}} = 848 \\
&v_{\text{Antipsychotics}} = 934 \\
&v_{\text{Antihistamines}} = 114 \\
&v_{\text{Corticosteroids}} = 1357 \\
&v_{\text{Beta Blockers}} = 156 \\
&v_{\text{Calcium Channel Blockers}} = 1780 \\
&v_{\text{ACE Inhibitors}} = 695 \\
&v_{\text{Angiotensin II Receptor Blockers}} = 405 \\
&v_{\text{Diuretics}} = 320 \\
&v_{\text{Statins}} = 320 \\
&v_{\text{Insulin}} = 1357 \\
&v_{\text{Anticoagulants}} = 1357 \\
&v_{\text{Antiepileptic Drugs}} = 405 \\
&v_{\text{Antiemetics}} = 998 \\
\end{align*}
\]

**Constraints:**

1. **Overall Stock Capacity:**
\[
\sum_{i} w_i x_i \leq 520
\]
where the $w_i$ are as follows (in file order):

\[
\begin{align*}
&w_{\text{NSAIDs}} = 913 \\
&w_{\text{Antirheumatic Drugs}} = 754 \\
&w_{\text{Acetic Acid Derivatives}} = 428 \\
&w_{\text{Antibiotics}} = 711 \\
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
\end{align*}
\]

2. **Nonnegativity and Integrality:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Full Model:**

\[
\begin{align*}
\max \quad & 250\,x_{\text{NSAIDs}} + 178\,x_{\text{Antirheumatic Drugs}} + 313\,x_{\text{Acetic Acid Derivatives}} + 301\,x_{\text{Antibiotics}} + 425\,x_{\text{Antiviral Drugs}} \\
& + 260\,x_{\text{Antifungal Agents}} + 848\,x_{\text{Antidepressants}} + 934\,x_{\text{Antipsychotics}} + 114\,x_{\text{Antihistamines}} + 1357\,x_{\text{Corticosteroids}} \\
& + 156\,x_{\text{Beta Blockers}} + 1780\,x_{\text{Calcium Channel Blockers}} + 695\,x_{\text{ACE Inhibitors}} + 405\,x_{\text{Angiotensin II Receptor Blockers}} \\
& + 320\,x_{\text{Diuretics}} + 320\,x_{\text{Statins}} + 1357\,x_{\text{Insulin}} + 1357\,x_{\text{Anticoagulants}} + 405\,x_{\text{Antiepileptic Drugs}} + 998\,x_{\text{Antiemetics}} \\
\text{s.t.} \quad & 913\,x_{\text{NSAIDs}} + 754\,x_{\text{Antirheumatic Drugs}} + 428\,x_{\text{Acetic Acid Derivatives}} + 711\,x_{\text{Antibiotics}} + 350\,x_{\text{Antiviral Drugs}} \\
& + 159\,x_{\text{Antifungal Agents}} + 353\,x_{\text{Antidepressants}} + 291\,x_{\text{Antipsychotics}} + 302\,x_{\text{Antihistamines}} + 50\,x_{\text{Corticosteroids}} \\
& + 250\,x_{\text{Beta Blockers}} + 178\,x_{\text{Calcium Channel Blockers}} + 313\,x_{\text{ACE Inhibitors}} + 378\,x_{\text{Angiotensin II Receptor Blockers}} \\
& + 94\,x_{\text{Diuretics}} + 97\,x_{\text{Statins}} + 470\,x_{\text{Insulin}} + 341\,x_{\text{Anticoagulants}} + 121\,x_{\text{Antiepileptic Drugs}} + 61\,x_{\text{Antiemetics}} \leq 520 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
\]