Retrieved data from "manager_project_costs.csv" (in source order, with exact identifiers and values):

**Document 1**
```json
{
  "record_view_count": "27",
  "Unnamed: 1": "MA",
  "archive_revision_number": "2",
  "P1": "3000",
  "archive_storage_medium": "Digital",
  "document_page_count": "6",
  "document_template_family": "Compact",
  "P2": "3200",
  "P3": "3100",
  "record_label_font": "Helvetica",
  "record_display_theme": "Amber",
  "archive_batch_number": "305"
}
```

**Document 2**
```json
{
  "record_view_count": "76",
  "Unnamed: 1": "MB",
  "archive_revision_number": "2",
  "P1": "2800",
  "archive_storage_medium": "Hybrid",
  "document_page_count": "8",
  "document_template_family": "Landscape",
  "P2": "3300",
  "P3": "2900",
  "record_label_font": "Calibri",
  "record_display_theme": "Amber",
  "archive_batch_number": "301"
}
```

**Document 3**
```json
{
  "record_view_count": "58",
  "Unnamed: 1": "MC",
  "archive_revision_number": "6",
  "P1": "2900",
  "archive_storage_medium": "Digital",
  "document_page_count": "4",
  "document_template_family": "Landscape",
  "P2": "3100",
  "P3": "3000",
  "record_label_font": "Calibri",
  "record_display_theme": "Azure",
  "archive_batch_number": "303"
}
```

**Relevant fields for model formulation (costs for each manager and project):**

| Manager | P1   | P2   | P3   |
|---------|------|------|------|
| MA      | 3000 | 3200 | 3100 |
| MB      | 2800 | 3300 | 2900 |
| MC      | 2900 | 3100 | 3000 |

All other fields are preserved above for reference.