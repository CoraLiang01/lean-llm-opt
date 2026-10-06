Here is the complete offer data for all available assignments, including worker_id, project_id, cost_cents, on_leave, worker_skill, and required_skill. Only workers with on_leave=0 are included, and only assignments where the worker's skill meets or exceeds the project's required_skill are listed.

**Worker and Project Data:**

| worker_id | on_leave | worker_skill  |
|-----------|----------|--------------|
| W00       | 0        | Junior       |
| W04       | 0        | Junior       |
| W06       | 0        | Expert       |
| W10       | 0        | Expert       |
| W11       | 0        | Senior       |
| W14       | 0        | Intermediate |
| W15       | 0        | Expert       |
| W19       | 0        | Junior       |
| W20       | 0        | Intermediate |

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

**Offer Data (filtered for eligibility):**

- **P00 (Intermediate):** Eligible: Intermediate, Senior, Expert
    - W06 (Expert): 1217
    - W10 (Expert): 147
    - W11 (Senior): 235
    - W15 (Expert): 722
    - W20 (Intermediate): 675

- **P01 (Intermediate):** Eligible: Intermediate, Senior, Expert
    - W06 (Expert): 425
    - W10 (Expert): 114
    - W14 (Intermediate): 1361
    - W15 (Expert): 0 (no offer)
    - W20 (Intermediate): 1055

- **P02 (Senior):** Eligible: Senior, Expert
    - W06 (Expert): 1320
    - W10 (Expert): 0 (no offer)
    - W11 (Senior): 1034
    - W15 (Expert): 577

- **P03 (Junior):** Eligible: Junior, Intermediate, Senior, Expert
    - W04 (Junior): 543
    - W06 (Expert): 1097
    - W10 (Expert): 998
    - W11 (Senior): 782
    - W14 (Intermediate): 379
    - W15 (Expert): 1062
    - W19 (Junior): 109
    - W20 (Intermediate): 703

- **P04 (Junior):** Eligible: Junior, Intermediate, Senior, Expert
    - W00 (Junior): 430
    - W04 (Junior): 205
    - W06 (Expert): 822
    - W10 (Expert): 1270
    - W11 (Senior): 556
    - W14 (Intermediate): 239
    - W15 (Expert): 1314
    - W19 (Junior): 1306

- **P05 (Junior):** Eligible: Junior, Intermediate, Senior, Expert
    - W00 (Junior): 1385
    - W04 (Junior): 1149
    - W06 (Expert): 221
    - W11 (Senior): 1018
    - W14 (Intermediate): 460
    - W15 (Expert): 528
    - W19 (Junior): 626

- **P06 (Junior):** Eligible: Junior, Intermediate, Senior, Expert
    - W00 (Junior): 260
    - W04 (Junior): 533
    - W06 (Expert): 496
    - W14 (Intermediate): 130
    - W15 (Expert): 835
    - W19 (Junior): 466
    - W20 (Intermediate): 893

- **P07 (Junior):** Eligible: Junior, Intermediate, Senior, Expert
    - W00 (Junior): 1107
    - W04 (Junior): 732
    - W06 (Expert): 908
    - W10 (Expert): 953
    - W11 (Senior): 1136
    - W15 (Expert): 918
    - W19 (Junior): 1365
    - W20 (Intermediate): 129

**Note:** W01 is excluded (on_leave=1).

---

**Summary Table (all eligible offers):**

| worker_id | worker_skill  | project_id | required_skill | cost_cents |
|-----------|--------------|------------|---------------|------------|
| W06       | Expert       | P00        | Intermediate  | 1217       |
| W10       | Expert       | P00        | Intermediate  | 147        |
| W11       | Senior       | P00        | Intermediate  | 235        |
| W15       | Expert       | P00        | Intermediate  | 722        |
| W20       | Intermediate | P00        | Intermediate  | 675        |
| W06       | Expert       | P01        | Intermediate  | 425        |
| W10       | Expert       | P01        | Intermediate  | 114        |
| W14       | Intermediate | P01        | Intermediate  | 1361       |
| W20       | Intermediate | P01        | Intermediate  | 1055       |
| W06       | Expert       | P02        | Senior        | 1320       |
| W11       | Senior       | P02        | Senior        | 1034       |
| W15       | Expert       | P02        | Senior        | 577        |
| W04       | Junior       | P03        | Junior        | 543        |
| W06       | Expert       | P03        | Junior        | 1097       |
| W10       | Expert       | P03        | Junior        | 998        |
| W11       | Senior       | P03        | Junior        | 782        |
| W14       | Intermediate | P03        | Junior        | 379        |
| W15       | Expert       | P03        | Junior        | 1062       |
| W19       | Junior       | P03        | Junior        | 109        |
| W20       | Intermediate | P03        | Junior        | 703        |
| W00       | Junior       | P04        | Junior        | 430        |
| W04       | Junior       | P04        | Junior        | 205        |
| W06       | Expert       | P04        | Junior        | 822        |
| W10       | Expert       | P04        | Junior        | 1270       |
| W11       | Senior       | P04        | Junior        | 556        |
| W14       | Intermediate | P04        | Junior        | 239        |
| W15       | Expert       | P04        | Junior        | 1314       |
| W19       | Junior       | P04        | Junior        | 1306       |
| W00       | Junior       | P05        | Junior        | 1385       |
| W04       | Junior       | P05        | Junior        | 1149       |
| W06       | Expert       | P05        | Junior        | 221        |
| W11       | Senior       | P05        | Junior        | 1018       |
| W14       | Intermediate | P05        | Junior        | 460        |
| W15       | Expert       | P05        | Junior        | 528        |
| W19       | Junior       | P05        | Junior        | 626        |
| W00       | Junior       | P06        | Junior        | 260        |
| W04       | Junior       | P06        | Junior        | 533        |
| W06       | Expert       | P06        | Junior        | 496        |
| W14       | Intermediate | P06        | Junior        | 130        |
| W15       | Expert       | P06        | Junior        | 835        |
| W19       | Junior       | P06        | Junior        | 466        |
| W20       | Intermediate | P06        | Junior        | 893        |
| W00       | Junior       | P07        | Junior        | 1107       |
| W04       | Junior       | P07        | Junior        | 732        |
| W06       | Expert       | P07        | Junior        | 908        |
| W10       | Expert       | P07        | Junior        | 953        |
| W11       | Senior       | P07        | Junior        | 1136       |
| W15       | Expert       | P07        | Junior        | 918        |
| W19       | Junior       | P07        | Junior        | 1365       |
| W20       | Intermediate | P07        | Junior        | 129        |

This is the full eligible offer list for the assignment problem as requested.