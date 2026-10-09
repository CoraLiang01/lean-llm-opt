##### Objective Function:

$\quad \max \sum_{i=1}^{21} v_i x_i$

where $x_i$ is the number of units of product $i$ to order, and $v_i$ is the benefit per unit of product $i$.

##### Constraints:

$\sum_{i=1}^{21} w_i x_i \leq 520$

$x_i \geq 0$ and integer, for all $i = 1, \ldots, 21$

##### Retrieved Information

{
  "capacity": 520,
  "products": [
    {
      "ProductName": "NSAIDs",
      "Value": 250,
      "Weight": 913
    },
    {
      "ProductName": "Antirheumatic Drugs",
      "Value": 178,
      "Weight": 754
    },
    {
      "ProductName": "Acetic Acid Derivatives",
      "Value": 313,
      "Weight": 428
    },
    {
      "ProductName": "Antibiotics",
      "Value": 301,
      "Weight": 711
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
      "Value": 934,
      "Weight": 291
    },
    {
      "ProductName": "Antihistamines",
      "Value": 114,
      "Weight": 302
    },
    {
      "ProductName": "Corticosteroids",
      "Value": 1357,
      "Weight": 50
    },
    {
      "ProductName": "Beta Blockers",
      "Value": 156,
      "Weight": 250
    },
    {
      "ProductName": "Calcium Channel Blockers",
      "Value": 1780,
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
      "Value": 320,
      "Weight": 97
    },
    {
      "ProductName": "Insulin",
      "Value": 1357,
      "Weight": 470
    },
    {
      "ProductName": "Anticoagulants",
      "Value": 1357,
      "Weight": 341
    },
    {
      "ProductName": "Antiepileptic Drugs",
      "Value": 405,
      "Weight": 121
    },
    {
      "ProductName": "Antiemetics",
      "Value": 998,
      "Weight": 61
    }
  ]
}

##### Parameter Vectors

Let the products be indexed in the order above ($i=1$ for NSAIDs, $i=2$ for Antirheumatic Drugs, ..., $i=21$ for Antiemetics):

- $v = [250, 178, 313, 301, 425, 260, 848, 934, 114, 1357, 156, 1780, 695, 405, 320, 320, 1357, 1357, 405, 998]$
- $w = [913, 754, 428, 711, 350, 159, 353, 291, 302, 50, 250, 178, 313, 378, 94, 97, 470, 341, 121, 61]$

##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1, \ldots, 21$ (number of units of each product to order)