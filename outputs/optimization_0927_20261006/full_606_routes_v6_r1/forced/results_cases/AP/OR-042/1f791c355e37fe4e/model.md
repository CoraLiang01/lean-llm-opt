##### Objective Function:

$\quad \max \sum_{i=1}^{20} v_i x_i$

where $x_i$ is the integer number of units of drug type $i$ to order, and $v_i$ is the benefit coefficient for drug type $i$.

##### Constraints:

$\sum_{i=1}^{20} w_i x_i \leq 4120$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,20\}$

##### Retrieved Information

{
  "capacity": 4120,
  "products": [
    {
      "ProductName": "NSAIDs",
      "Value": 585,
      "Weight": 50
    },
    {
      "ProductName": "Antirheumatic Drugs",
      "Value": 557,
      "Weight": 329
    },
    {
      "ProductName": "Acetic Acid Derivatives",
      "Value": 963,
      "Weight": 410
    },
    {
      "ProductName": "Antibiotics",
      "Value": 301,
      "Weight": 452
    },
    {
      "ProductName": "Antiviral Drugs",
      "Value": 425,
      "Weight": 350
    },
    {
      "ProductName": "Antifungal Agents",
      "Value": 260,
      "Weight": 159
    },
    {
      "ProductName": "Antidepressants",
      "Value": 848,
      "Weight": 353
    },
    {
      "ProductName": "Antipsychotics",
      "Value": 461,
      "Weight": 291
    },
    {
      "ProductName": "Antihistamines",
      "Value": 840,
      "Weight": 302
    },
    {
      "ProductName": "Corticosteroids",
      "Value": 999,
      "Weight": 50
    },
    {
      "ProductName": "Beta Blockers",
      "Value": 392,
      "Weight": 250
    },
    {
      "ProductName": "Calcium Channel Blockers",
      "Value": 874,
      "Weight": 178
    },
    {
      "ProductName": "ACE Inhibitors",
      "Value": 695,
      "Weight": 313
    },
    {
      "ProductName": "Angiotensin II Receptor Blockers",
      "Value": 405,
      "Weight": 378
    },
    {
      "ProductName": "Diuretics",
      "Value": 320,
      "Weight": 94
    },
    {
      "ProductName": "Statins",
      "Value": 913,
      "Weight": 97
    },
    {
      "ProductName": "Insulin",
      "Value": 754,
      "Weight": 470
    },
    {
      "ProductName": "Anticoagulants",
      "Value": 428,
      "Weight": 341
    },
    {
      "ProductName": "Antiepileptic Drugs",
      "Value": 711,
      "Weight": 121
    },
    {
      "ProductName": "Antiemetics",
      "Value": 998,
      "Weight": 61
    }
  ]
}

##### Parameter Table

| $i$ | Product Name                        | $v_i$ (Benefit) | $w_i$ (Weight) |
|-----|-------------------------------------|-----------------|---------------|
| 1   | NSAIDs                             | 585             | 50            |
| 2   | Antirheumatic Drugs                | 557             | 329           |
| 3   | Acetic Acid Derivatives            | 963             | 410           |
| 4   | Antibiotics                        | 301             | 452           |
| 5   | Antiviral Drugs                    | 425             | 350           |
| 6   | Antifungal Agents                  | 260             | 159           |
| 7   | Antidepressants                    | 848             | 353           |
| 8   | Antipsychotics                     | 461             | 291           |
| 9   | Antihistamines                     | 840             | 302           |
| 10  | Corticosteroids                    | 999             | 50            |
| 11  | Beta Blockers                      | 392             | 250           |
| 12  | Calcium Channel Blockers           | 874             | 178           |
| 13  | ACE Inhibitors                     | 695             | 313           |
| 14  | Angiotensin II Receptor Blockers   | 405             | 378           |
| 15  | Diuretics                          | 320             | 94            |
| 16  | Statins                            | 913             | 97            |
| 17  | Insulin                            | 754             | 470           |
| 18  | Anticoagulants                     | 428             | 341           |
| 19  | Antiepileptic Drugs                | 711             | 121           |
| 20  | Antiemetics                        | 998             | 61            |