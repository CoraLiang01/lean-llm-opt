Here is the complete offer data for all available worker-project pairings, including each worker's leave status and skill, and each project's required skill. Only workers with on_leave=0 are eligible. The skill hierarchy is: Junior < Intermediate < Senior < Expert, so a worker can only be assigned to a project if their skill is at least the required_skill.

**Worker Data (excluding on_leave=1):**
| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W00       | Junior      | 0        |
| W04       | Junior      | 0        |
| W06       | Expert      | 0        |
| W10       | Expert      | 0        |
| W11       | Senior      | 0        |
| W14       | Intermediate| 0        |
| W15       | Expert      | 0        |
| W19       | Junior      | 0        |
| W20       | Intermediate| 0        |

**Project Data:**
| project_id | required_skill |
|------------|---------------|
| P00        | Intermediate  |
| P01        | Intermediate  |
| P02        | Senior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Junior        |
| P06        | Junior        |
| P07        | Junior        |

**Offer Data (all available pairings, with worker skill, on_leave, and project required_skill):**

| worker_id | worker_skill | on_leave | project_id | required_skill | cost_cents |
|-----------|-------------|----------|------------|---------------|------------|
| W00       | Junior      | 0        | P01        | Intermediate  | 17         |
| W00       | Junior      | 0        | P04        | Junior        | 430        |
| W00       | Junior      | 0        | P05        | Junior        | 1385       |
| W00       | Junior      | 0        | P06        | Junior        | 260        |
| W00       | Junior      | 0        | P07        | Junior        | 1107       |
| W04       | Junior      | 0        | P00        | Intermediate  | 4          |
| W04       | Junior      | 0        | P01        | Intermediate  | 8          |
| W04       | Junior      | 0        | P02        | Senior        | 16         |
| W04       | Junior      | 0        | P03        | Junior        | 543        |
| W04       | Junior      | 0        | P04        | Junior        | 205        |
| W04       | Junior      | 0        | P05        | Junior        | 1149       |
| W04       | Junior      | 0        | P06        | Junior        | 533        |
| W04       | Junior      | 0        | P07        | Junior        | 732        |
| W06       | Expert      | 0        | P00        | Intermediate  | 1217       |
| W06       | Expert      | 0        | P01        | Intermediate  | 425        |
| W06       | Expert      | 0        | P02        | Senior        | 1320       |
| W06       | Expert      | 0        | P03        | Junior        | 1097       |
| W06       | Expert      | 0        | P04        | Junior        | 822        |
| W06       | Expert      | 0        | P05        | Junior        | 221        |
| W06       | Expert      | 0        | P06        | Junior        | 496        |
| W06       | Expert      | 0        | P07        | Junior        | 908        |
| W10       | Expert      | 0        | P00        | Intermediate  | 147        |
| W10       | Expert      | 0        | P01        | Intermediate  | 114        |
| W10       | Expert      | 0        | P03        | Junior        | 998        |
| W10       | Expert      | 0        | P04        | Junior        | 1270       |
| W10       | Expert      | 0        | P07        | Junior        | 953        |
| W11       | Senior      | 0        | P00        | Intermediate  | 235        |
| W11       | Senior      | 0        | P02        | Senior        | 1034       |
| W11       | Senior      | 0        | P03        | Junior        | 782        |
| W11       | Senior      | 0        | P04        | Junior        | 556        |
| W11       | Senior      | 0        | P05        | Junior        | 1018       |
| W11       | Senior      | 0        | P07        | Junior        | 1136       |
| W14       | Intermediate| 0        | P01        | Intermediate  | 1361       |
| W14       | Intermediate| 0        | P03        | Junior        | 379        |
| W14       | Intermediate| 0        | P04        | Junior        | 239        |
| W14       | Intermediate| 0        | P05        | Junior        | 460        |
| W14       | Intermediate| 0        | P06        | Junior        | 130        |
| W15       | Expert      | 0        | P00        | Intermediate  | 722        |
| W15       | Expert      | 0        | P02        | Senior        | 577        |
| W15       | Expert      | 0        | P03        | Junior        | 1062       |
| W15       | Expert      | 0        | P04        | Junior        | 1314       |
| W15       | Expert      | 0        | P05        | Junior        | 528        |
| W15       | Expert      | 0        | P06        | Junior        | 835        |
| W15       | Expert      | 0        | P07        | Junior        | 918        |
| W19       | Junior      | 0        | P00        | Intermediate  | 5          |
| W19       | Junior      | 0        | P02        | Senior        | 9          |
| W19       | Junior      | 0        | P03        | Junior        | 109        |
| W19       | Junior      | 0        | P04        | Junior        | 1306       |
| W19       | Junior      | 0        | P05        | Junior        | 626        |
| W19       | Junior      | 0        | P06        | Junior        | 466        |
| W19       | Junior      | 0        | P07        | Junior        | 1365       |
| W20       | Intermediate| 0        | P00        | Intermediate  | 675        |
| W20       | Intermediate| 0        | P01        | Intermediate  | 1055       |
| W20       | Intermediate| 0        | P02        | Senior        | 8          |
| W20       | Intermediate| 0        | P03        | Junior        | 703        |
| W20       | Intermediate| 0        | P06        | Junior        | 893        |
| W20       | Intermediate| 0        | P07        | Junior        | 129        |

**Note:**  
- Only offers where worker_skill ≥ required_skill are valid.
- Only workers with on_leave=0 are eligible.

This is the complete data needed for the assignment and cost minimization.