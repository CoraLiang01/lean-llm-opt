##### Objective Function:

$\quad \min \sum_{i=1}^3 \sum_{j=1}^3 c_{ij} x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to complete project $j$, and $x_{ij}$ is a binary variable indicating whether manager $i$ is assigned to project $j$.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j=1}^3 x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i=1}^3 x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 3. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

##### Retrieved Information

{
  "cost": {
    "MA": {
      "P1": 3000,
      "P2": 3200,
      "P3": 3100
    },
    "MB": {
      "P1": 2800,
      "P2": 3300,
      "P3": 2900
    },
    "MC": {
      "P1": 2900,
      "P2": 3100,
      "P3": 3000
    }
  },
  "managers": [
    "MA",
    "MB",
    "MC"
  ],
  "projects": [
    "P1",
    "P2",
    "P3"
  ],
  "full_source_rows": [
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
    },
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
    },
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
  ]
}