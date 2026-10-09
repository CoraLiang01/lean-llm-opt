@lru_cache(maxsize=1)
def _get_classifier_retriever():
    frame = loto_reference_frame()
    documents = [
        Document(
            page_content="\n".join(
                f"{column}: {str(example[column]).strip()}"
                for column in ("prompt", "Type")
            ),
            metadata={
                "source": str(PROJECT_ROOT / "Large_Scale_Or_Files/RAG_Examples_All.csv"),
                "row": row_number,
            },
        )
        for row_number, example in enumerate(frame.to_dict("records"))
    ]
    if len(documents) < 5:
        raise ValueError("Classification reference requires at least five labeled examples")
    return FAISS.from_documents(documents, make_embeddings()).as_retriever(
        search_kwargs={"k": 5}
    )


@lru_cache(maxsize=1)
def _load_rag_table():
    """Read the filtered examples once, preserving every field as text."""
    frame = loto_reference_frame().copy()
    required = {"prompt", "Data_address", "Related", "Required Data", "Label", "Code", "Type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RAG example CSV is missing columns: {sorted(missing)}")
    frame["Type"] = frame["Type"].map(normalize_route)
    return frame


def load_rag_examples(route):
    """Return one canonical route; UFLP maps to FLP and Mixture to Others."""
    frame = _load_rag_table()
    examples = frame.loc[frame["Type"].eq(normalize_route(route))].copy()
    return examples


def retrieve_rag_examples(route, query, k=1):
    """Retrieve structured example dictionaries, never parse CSVLoader display text."""
    if k < 1:
        return []
    route = assert_loto_route_allowed(route)
    examples = load_rag_examples(route)
    if examples.empty:
        return []
    documents = _get_rag_store(route).similarity_search(
        str(query), k=min(k, len(examples)),
    )
    return [json.loads(document.metadata["example_json"]) for document in documents]



def filter_loto_query_only_documents(documents):
    """Scope the separate query-only library by its original problem-type field."""
    target = require_loto_fold()["held_out_type"]
    def semantic_type(document):
        match = re.search(r"^problem type:\s*(.*)$", document.page_content, re.MULTILINE | re.I)
        text = match.group(1).casefold() if match else "others"
        if "network revenue" in text:
            return "NRM"
        if "resource allocation" in text:
            return "RA"
        if "assignment" in text:
            return "AP"
        if "facility location" in text:
            return "FLP"
        if "transportation" in text:
            return "TP"
        return "Others"
    return [document for document in documents if semantic_type(document) != target]


def filter_loto_query_only_triggers(text):
    """Remove target-type fixed demonstrations while retaining generic contracts."""
    target = require_loto_fold()["held_out_type"]
    # Explicit structural types of the three fixed baseline demonstrations.
    trigger_types = {1: "RA", 2: "Mixture", 3: "Mixture"}
    for number, label in trigger_types.items():
        if label == target:
            text = re.sub(r"\(" + str(number) + r"\).*?(?=\(\d\)|USER QUESTION:)",
                          "", text, flags=re.DOTALL)
    return text
