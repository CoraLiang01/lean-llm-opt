Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given below.

**Parameters:**

- $b_i$: Value (benefit) per unit of product $i$
- $w_i$: Weight (stock usage) per unit of product $i$
- $C$: Total stock capacity

**Data (in source order):**

| $i$ | ProductName                           | $b_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------------------------|---------------|---------------|
| 1   | NSAIDs                                | 250           | 913           |
| 2   | Antirheumatic Drugs                   | 178           | 754           |
| 3   | Acetic Acid Derivatives               | 313           | 428           |
| 4   | Antibiotics                           | 301           | 711           |
| 5   | Antiviral Drugs                       | 425           | 350           |
| 6   | Antifungal Agents                     | 260           | 159           |
| 7   | Antidepressants                       | 848           | 353           |
| 8   | Antipsychotics                        | 934           | 291           |
| 9   | Antihistamines                        | 114           | 302           |
| 10  | Corticosteroids                       | 1357          | 50            |
| 11  | Beta Blockers                         | 156           | 250           |
| 12  | Calcium Channel Blockers              | 1780          | 178           |
| 13  | ACE Inhibitors                        | 695           | 313           |
| 14  | Angiotensin II Receptor Blockers      | 405           | 378           |
| 15  | Diuretics                             | 320           | 94            |
| 16  | Statins                               | 320           | 97            |
| 17  | Insulin                               | 1357          | 470           |
| 18  | Anticoagulants                        | 1357          | 341           |
| 19  | Antiepileptic Drugs                   | 405           | 121           |
| 20  | Antiemetics                           | 998           | 61            |

Total stock capacity: $C = 520$

---

**Mathematical Model:**

Maximize total benefit:
$$
\max \sum_{i=1}^{20} b_i x_i
$$

Subject to the overall stock capacity:
$$
\sum_{i=1}^{20} w_i x_i \leq 520
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 20
$$

Where the parameters $b_i$ and $w_i$ are as listed above for each product.