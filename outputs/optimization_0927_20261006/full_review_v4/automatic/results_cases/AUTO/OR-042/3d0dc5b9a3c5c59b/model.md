Let $x_i$ be the number of units of drug type $i$ to order daily, where $i$ indexes the following drug types in the order given:

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

Let $v_i$ be the benefit coefficient and $w_i$ the weight per unit for drug type $i$ as given below.

#### Parameters

| $i$ | ProductName                          | $v_i$ (Value) | $w_i$ (Weight) |
|-----|--------------------------------------|---------------|---------------|
| 1   | NSAIDs                              | 585           | 50            |
| 2   | Antirheumatic Drugs                 | 557           | 329           |
| 3   | Acetic Acid Derivatives             | 963           | 410           |
| 4   | Antibiotics                         | 301           | 452           |
| 5   | Antiviral Drugs                     | 425           | 350           |
| 6   | Antifungal Agents                   | 260           | 159           |
| 7   | Antidepressants                     | 848           | 353           |
| 8   | Antipsychotics                      | 461           | 291           |
| 9   | Antihistamines                      | 840           | 302           |
| 10  | Corticosteroids                     | 999           | 50            |
| 11  | Beta Blockers                       | 392           | 250           |
| 12  | Calcium Channel Blockers            | 874           | 178           |
| 13  | ACE Inhibitors                      | 695           | 313           |
| 14  | Angiotensin II Receptor Blockers    | 405           | 378           |
| 15  | Diuretics                           | 320           | 94            |
| 16  | Statins                             | 913           | 97            |
| 17  | Insulin                             | 754           | 470           |
| 18  | Anticoagulants                      | 428           | 341           |
| 19  | Antiepileptic Drugs                 | 711           | 121           |
| 20  | Antiemetics                         | 998           | 61            |

Total inventory capacity: $C = 4120$

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{20} v_i x_i
\]

**Subject to:**
\[
\sum_{i=1}^{20} w_i x_i \leq 4120
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
\]

**Where:**

- $x_i$ = number of units of drug type $i$ to order daily (integer, $\geq 0$)
- $v_i$ = benefit coefficient for drug type $i$ (see table above)
- $w_i$ = weight per unit for drug type $i$ (see table above)
- $C = 4120$ = total inventory capacity (weight units)

**All coefficients and identifiers are as retrieved and in original order.**