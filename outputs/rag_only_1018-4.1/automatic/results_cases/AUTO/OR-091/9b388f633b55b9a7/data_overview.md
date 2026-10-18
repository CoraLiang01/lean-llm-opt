File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 42
Columns: ['course_id', 'course_name', 'discipline', 'credits', 'interest_points']
Parsed column types: {'course_id': 'object', 'course_name': 'object', 'discipline': 'object', 'credits': 'int64', 'interest_points': 'int64'}
Preview only (first 10 rows):
course_id            course_name  discipline credits interest_points
      C01         World Classics  Literature       4              58
      C02          Modern Poetry  Literature       3              55
      C03 Comparative Literature  Literature       4              60
      C04        Literary Theory  Literature       4              62
      C05      Narrative Studies  Literature       4              57
      C06    Shakespeare Studies  Literature       4              59
      C07       Creative Writing  Literature       3              63
      C08            Calculus II Mathematics       5              70
      C09         Linear Algebra Mathematics       4              78
      C10            Probability Mathematics       4              75
Full-file column statistics: {"course_id": {"missing": 0, "unique_nonempty": 42}, "course_name": {"missing": 0, "unique_nonempty": 42}, "discipline": {"missing": 0, "unique_nonempty": 6}, "credits": {"missing": 0, "unique_nonempty": 3, "numeric_range": [3.0, 5.0]}, "interest_points": {"missing": 0, "unique_nonempty": 32, "numeric_range": [55.0, 95.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "chemistry", "matching_columns": [{"column": "course_name", "exact": 0, "prefix": 0, "contains": 7, "examples": ["Organic Chemistry I", "Inorganic Chemistry", "Physical Chemistry"]}, {"column": "discipline", "exact": 7, "prefix": 7, "contains": 7, "examples": ["Chemistry"]}], "exact_matching_columns": 1}, {"term": "computer science", "matching_columns": [{"column": "discipline", "exact": 7, "prefix": 7, "contains": 7, "examples": ["Computer Science"]}], "exact_matching_columns": 1}, {"term": "credits", "matching_columns": [], "exact_matching_columns": 0}, {"term": "discipline", "matching_columns": [], "exact_matching_columns": 0}, {"term": "interest points", "matching_columns": [], "exact_matching_columns": 0}, {"term": "literature", "matching_columns": [{"column": "course_name", "exact": 0, "prefix": 0, "contains": 1, "examples": ["Comparative Literature"]}, {"column": "discipline", "exact": 7, "prefix": 7, "contains": 7, "examples": ["Literature"]}], "exact_matching_columns": 1}, {"term": "mathematics", "matching_columns": [{"column": "course_name", "exact": 0, "prefix": 0, "contains": 1, "examples": ["Discrete Mathematics"]}, {"column": "discipline", "exact": 7, "prefix": 7, "contains": 7, "examples": ["Mathematics"]}], "exact_matching_columns": 1}, {"term": "operations research", "matching_columns": [{"column": "course_name", "exact": 0, "prefix": 1, "contains": 1, "examples": ["Operations Research: Linear Programming"]}, {"column": "discipline", "exact": 7, "prefix": 7, "contains": 7, "examples": ["Operations Research"]}], "exact_matching_columns": 1}, {"term": "physics", "matching_columns": [{"column": "course_name", "exact": 0, "prefix": 0, "contains": 3, "examples": ["Quantum Physics", "Statistical Physics", "Solid State Physics"]}, {"column": "discipline", "exact": 7, "prefix": 7, "contains": 7, "examples": ["Physics"]}], "exact_matching_columns": 1}]