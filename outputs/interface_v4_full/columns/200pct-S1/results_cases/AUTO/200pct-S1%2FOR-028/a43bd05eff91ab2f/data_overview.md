**PotentialWarehouses_Costs.csv**  
| Row | Warehouse (i) | archive_revision_number | document_page_count | Opening Cost (fi) | archive_storage_medium | record_display_theme | document_template_family | record_view_count | Capacity (units) |
|-----|---------------|------------------------|--------------------|-------------------|-----------------------|---------------------|-------------------------|-------------------|------------------|
| 1   | 1             | 1                      | 6                  | 3000              | Digital               | Olive               | Landscape               | 76                | 180              |
| 2   | 2             | 6                      | 6                  | 3200              | Hybrid                | Olive               | Compact                 | 76                | 160              |
| 3   | 3             | 4                      | 16                 | 3100              | Digital               | Olive               | Compact                 | 76                | 200              |
| 4   | 4             | 1                      | 8                  | 2800              | Digital               | Olive               | Landscape               | 76                | 150              |
| 5   | 5             | 1                      | 6                  | 3500              | Hybrid                | Azure               | Landscape               | 12                | 170              |
| 6   | 6             | 1                      | 2                  | 2700              | Paper                 | Amber               | Standard                | 43                | 190              |
| 7   | 7             | 2                      | 6                  | 2900              | Hybrid                | Amber               | Standard                | 12                | 160              |
| 8   | 8             | 1                      | 16                 | 3050              | Hybrid                | Amber               | Standard                | 58                | 175              |
| 9   | 9             | 2                      | 4                  | 3100              | Digital               | Azure               | Landscape               | 58                | 170              |
| 10  | 10            | 5                      | 6                  | 2200              | Digital               | Amber               | Compact                 | 76                | 180              |
| 11  | 11            | 6                      | 12                 | 2890              | Paper                 | Olive               | Standard                | 76                | 190              |

---

**Stores_Demands.csv**  
| Row | Store (j) | archive_revision_number | record_view_count | archive_batch_number | Demand (units, dj) | document_page_count |
|-----|-----------|------------------------|-------------------|---------------------|--------------------|--------------------|
| 1   | 1         | 6                      | 76                | 305                 | 30                 | 8                  |
| 2   | 2         | 5                      | 27                | 301                 | 40                 | 12                 |
| 3   | 3         | 1                      | 27                | 304                 | 20                 | 12                 |
| 4   | 4         | 1                      | 58                | 305                 | 35                 | 4                  |
| 5   | 5         | 3                      | 43                | 304                 | 20                 | 4                  |
| 6   | 6         | 6                      | 76                | 304                 | 25                 | 16                 |
| 7   | 7         | 1                      | 58                | 304                 | 45                 | 12                 |
| 8   | 8         | 6                      | 91                | 301                 | 38                 | 8                  |
| 9   | 9         | 2                      | 27                | 301                 | 32                 | 4                  |
| 10  | 10        | 2                      | 43                | 302                 | 41                 | 16                 |
| 11  | 11        | 4                      | 27                | 302                 | 44                 | 16                 |

---

**TransportationCost.csv**  
Each row below corresponds to a warehouse (i), each column to a store (j).  
The cost c_ij is the transportation cost from warehouse i to store j.

| Warehouse (i) | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 | Source Row |
|---------------|----|----|----|----|----|----|----|----|----|-----|------|------------|
| 1             | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15   | archive_box_number: 18, archive_revision_number: 5 |
| 2             | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16   | archive_box_number: 39, archive_revision_number: 5 |
| 3             | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17   | archive_box_number: 25, archive_revision_number: 1 |
| 4             | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18   | archive_box_number: 18, archive_revision_number: 4 |
| 5             | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17   | archive_box_number: 32, archive_revision_number: 4 |
| 6             | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19   | archive_box_number: 11, archive_revision_number: 1 |
| 7             | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14   | archive_box_number: 11, archive_revision_number: 6 |
| 8             | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18   | archive_box_number: 25, archive_revision_number: 4 |
| 9             | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18   | archive_box_number: 39, archive_revision_number: 1 |
| 10            | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19   | archive_box_number: 18, archive_revision_number: 3 |
| 11            | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13   | archive_box_number: 32, archive_revision_number: 5 |

- W1 = Store 1, W2 = Store 2, ..., W11 = Store 11
- All source row positions and archive identifiers are preserved as requested.

---

**Summary of preserved axes and identifiers:**
- Facility (Warehouse) IDs: 1–11, with Opening Cost (fi) and Capacity (units) per warehouse.
- Customer (Store) IDs: 1–11, with Demand (units, dj) per store.
- Transportation cost matrix c_ij: Warehouse i (rows) to Store j (columns), with all source row and column IDs and archive metadata retained.
- No extra axes or inferred products; all data is as in the original sources.