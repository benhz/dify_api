from enum import StrEnum


class IndexType(StrEnum):
    PARAGRAPH_INDEX = "text_model"
    QA_INDEX = "qa_model"
    PARENT_CHILD_INDEX = "hierarchical_model"
    SEMANTIC_INDEX = "semantic_model"
