##### Decision Variables

$x_i \geq 0$: Number of units of drug product $i$ to order each day, for each $i \in P$ (continuous).

##### Parameters

Let $P$ be the set of drug products:

\[
P = \{
\text{NSAIDs},
\text{Antirheumatic Drugs},
\text{Acetic Acid Derivatives},
\text{Antibiotics},
\text{Antiviral Drugs},
\text{Antifungal Agents},
\text{Antidepressants},
\text{Antipsychotics},
\text{Antihistamines},
\text{Corticosteroids},
\text{Beta Blockers},
\text{Calcium Channel Blockers},
\text{ACE Inhibitors},
\text{Angiotensin II Receptor Blockers},
\text{Diuretics},
\text{Statins},
\text{Insulin},
\text{Anticoagulants},
\text{Antiepileptic Drugs},
\text{Antiemetics}
\}
\]

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

Let $C$ be the overall stock capacity:

\[
C = 520
\]

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
\max \quad & 250x_{\text{NSAIDs}} + 178x_{\text{Antirheumatic Drugs}} + 313x_{\text{Acetic Acid Derivatives}} + 301x_{\text{Antibiotics}} + 425x_{\text{Antiviral Drugs}} \\
& + 260x_{\text{Antifungal Agents}} + 848x_{\text{Antidepressants}} + 934x_{\text{Antipsychotics}} + 114x_{\text{Antihistamines}} + 1357x_{\text{Corticosteroids}} \\
& + 156x_{\text{Beta Blockers}} + 1780x_{\text{Calcium Channel Blockers}} + 695x_{\text{ACE Inhibitors}} + 405x_{\text{Angiotensin II Receptor Blockers}} \\
& + 320x_{\text{Diuretics}} + 320x_{\text{Statins}} + 1357x_{\text{Insulin}} + 1357x_{\text{Anticoagulants}} + 405x_{\text{Antiepileptic Drugs}} + 998x_{\text{Antiemetics}} \\
\text{s.t.} \quad & 913x_{\text{NSAIDs}} + 754x_{\text{Antirheumatic Drugs}} + 428x_{\text{Acetic Acid Derivatives}} + 711x_{\text{Antibiotics}} + 350x_{\text{Antiviral Drugs}} \\
& + 159x_{\text{Antifungal Agents}} + 353x_{\text{Antidepressants}} + 291x_{\text{Antipsychotics}} + 302x_{\text{Antihistamines}} + 50x_{\text{Corticosteroids}} \\
& + 250x_{\text{Beta Blockers}} + 178x_{\text{Calcium Channel Blockers}} + 313x_{\text{ACE Inhibitors}} + 378x_{\text{Angiotensin II Receptor Blockers}} \\
& + 94x_{\text{Diuretics}} + 97x_{\text{Statins}} + 470x_{\text{Insulin}} + 341x_{\text{Anticoagulants}} + 121x_{\text{Antiepileptic Drugs}} + 61x_{\text{Antiemetics}} \leq 520 \\
& x_i \geq 0 \quad \forall i \in P
\end{align*}
\]