Let the decision variables be:
x_i = number of units of drug type i to order daily, for each drug type i listed below. All x_i are nonnegative integers.

Data:
Let the drug types be indexed in the order given in products.csv:

| i | ProductName                        | Value | Weight |
|---|------------------------------------|-------|--------|
| 1 | NSAIDs                             | 585   | 50     |
| 2 | Antirheumatic Drugs                | 557   | 329    |
| 3 | Acetic Acid Derivatives            | 963   | 410    |
| 4 | Antibiotics                        | 301   | 452    |
| 5 | Antiviral Drugs                    | 425   | 350    |
| 6 | Antifungal Agents                  | 260   | 159    |
| 7 | Antidepressants                    | 848   | 353    |
| 8 | Antipsychotics                     | 461   | 291    |
| 9 | Antihistamines                     | 840   | 302    |
|10 | Corticosteroids                    | 999   | 50     |
|11 | Beta Blockers                      | 392   | 250    |
|12 | Calcium Channel Blockers           | 874   | 178    |
|13 | ACE Inhibitors                     | 695   | 313    |
|14 | Angiotensin II Receptor Blockers   | 405   | 378    |
|15 | Diuretics                          | 320   | 94     |
|16 | Statins                            | 913   | 97     |
|17 | Insulin                            | 754   | 470    |
|18 | Anticoagulants                     | 428   | 341    |
|19 | Antiepileptic Drugs                | 711   | 121    |
|20 | Antiemetics                        | 998   | 61     |

The overall inventory capacity is 4120 (from capacity.csv).

Model:

Variables:
For i = 1,...,20,
 x_i ∈ {0, 1, 2, ...}

Objective:
Maximize total benefit:
 maximize 585 x₁ + 557 x₂ + 963 x₃ + 301 x₄ + 425 x₅ + 260 x₆ + 848 x₇ + 461 x₈ + 840 x₉ + 999 x₁₀ + 392 x₁₁ + 874 x₁₂ + 695 x₁₃ + 405 x₁₄ + 320 x₁₅ + 913 x₁₆ + 754 x₁₇ + 428 x₁₈ + 711 x₁₉ + 998 x₂₀

Subject to:
Total weight constraint:
 50 x₁ + 329 x₂ + 410 x₃ + 452 x₄ + 350 x₅ + 159 x₆ + 353 x₇ + 291 x₈ + 302 x₉ + 50 x₁₀ + 250 x₁₁ + 178 x₁₂ + 313 x₁₃ + 378 x₁₄ + 94 x₁₅ + 97 x₁₆ + 470 x₁₇ + 341 x₁₈ + 121 x₁₉ + 61 x₂₀ ≤ 4120

x_i ∈ {0, 1, 2, ...} for all i = 1,...,20

Where:
x₁ = units of NSAIDs
x₂ = units of Antirheumatic Drugs
x₃ = units of Acetic Acid Derivatives
x₄ = units of Antibiotics
x₅ = units of Antiviral Drugs
x₆ = units of Antifungal Agents
x₇ = units of Antidepressants
x₈ = units of Antipsychotics
x₉ = units of Antihistamines
x₁₀ = units of Corticosteroids
x₁₁ = units of Beta Blockers
x₁₂ = units of Calcium Channel Blockers
x₁₃ = units of ACE Inhibitors
x₁₄ = units of Angiotensin II Receptor Blockers
x₁₅ = units of Diuretics
x₁₆ = units of Statins
x₁₇ = units of Insulin
x₁₈ = units of Anticoagulants
x₁₉ = units of Antiepileptic Drugs
x₂₀ = units of Antiemetics

This is a complete integer optimization model for the pharmacy's restocking problem.