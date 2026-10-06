**Retrieved Data**

---

### 1. Demand Data (`demand.csv`)

| Customer | Demand | Source Row |
|----------|--------|------------|
| C1       | 1083   | 1          |
| C2       | 776    | 2          |
| C3       | 16214  | 3          |

---

### 2. Fixed Cost Data (`fixed_cost.csv`)

| Facility (Warehouse) | FixedCost | Source Row |
|----------------------|-----------|------------|
| S1                   | 102.33    | 4          |
| S2                   | 94.92     | 5          |
| S3                   | 91.83     | 6          |

---

### 3. Transportation Cost Matrix (`transportation_costs.csv`)

**Source Row 7 (archive_batch_number: 301, archive_revision_number: 4, document_template_family: Standard, record_label_font: Helvetica, archive_storage_medium: Paper, record_display_theme: Olive, document_page_count: 6):**

| Facility (Warehouse) | C1      | C2    | C3   | Source Row |
|----------------------|---------|-------|------|------------|
| S1                   | 1506.22 | 70.9  | 8.44 | 7          |

**Source Row 8 (archive_batch_number: 303, archive_revision_number: 4, document_template_family: Compact, record_label_font: Calibri, archive_storage_medium: Paper, record_display_theme: Azure, document_page_count: 8):**

| Facility (Warehouse) | C1      | C2      | C3     | Source Row |
|----------------------|---------|---------|--------|------------|
| S2                   | 1732.65 | 1780.72 | 567.44 | 8          |

**Source Row 9 (archive_batch_number: 302, archive_revision_number: 1, document_template_family: Compact, record_label_font: Helvetica, archive_storage_medium: Digital, record_display_theme: Amber, document_page_count: 4):**

| Facility (Warehouse) | C1     | C2     | C3    | Source Row |
|----------------------|--------|--------|-------|------------|
| S3                   | 115.66 | 100.76 | 64.68 | 9          |

---

### 4. Facility/Customer/Cost Matrix Axis and Shape

- **Facilities (Warehouses):** S1, S2, S3
- **Customers (Musicians/Bands):** C1, C2, C3
- **Cost Matrix Shape:** 3 (facilities) × 3 (customers)
- **All axes and identifiers preserved as in source.**

---

### 5. Capacity Data

- **No explicit capacity data present for any facility.** (Unresolved evidence, not zero.)

---

**Summary Table (for modeling):**

| Facility | FixedCost | C1 (cost) | C2 (cost) | C3 (cost) | Source Row (cost) | Source Row (fixed) |
|----------|-----------|-----------|-----------|-----------|-------------------|--------------------|
| S1       | 102.33    | 1506.22   | 70.9      | 8.44      | 7                 | 4                  |
| S2       | 94.92     | 1732.65   | 1780.72   | 567.44    | 8                 | 5                  |
| S3       | 91.83     | 115.66    | 100.76    | 64.68     | 9                 | 6                  |

| Customer | Demand | Source Row |
|----------|--------|------------|
| C1       | 1083   | 1          |
| C2       | 776    | 2          |
| C3       | 16214  | 3          |

---

**All identifiers, values, and source-row positions are preserved. No transposition, truncation, or inference beyond the provided data.**