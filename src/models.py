"""Small validated records shared by extraction, generation, and downloads."""

from typing import Literal, Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Annotated type strictly mapping to non-empty strings
Nonempty = Annotated[str, Field(min_length=1)]

class Record(BaseModel):
    # Forbid unexpected kwargs strictly to avoid dropping generated fields silently
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class RequirementDraft(Record):
    statement: str = Field(min_length=1, max_length=800)
    source_quote: str = Field(min_length=1, max_length=1000)
    source_kind: Literal["text", "visual"]
    measurable: bool

class RequirementBatch(Record):
    requirements: list[RequirementDraft]

class Requirement(RequirementDraft):
    id: str
    page: int

class CitationDraft(Record):
    reference_id: str
    quote: str = Field(min_length=1)

class TestDraft(Record):
    title: str = Field(min_length=1)
    status: Literal["draft", "insufficient_information"]
    preconditions: list[Nonempty]
    steps: list[Nonempty]
    pass_criteria: str | None
    fail_criteria: str | None
    missing_information: list[Nonempty]
    citations: list[CitationDraft]

    @model_validator(mode="after")
    def validate_decision(self):
        # Enforce business logic rules dynamically
        if self.status == "draft":
            # A draft test MUST be executable and measurable
            if not self.steps or not self.pass_criteria or not self.fail_criteria or not self.citations:
                raise ValueError("draft requires steps, pass/fail criteria, and citations")
            if self.missing_information:
                raise ValueError("unresolved information requires abstention")

        elif not self.missing_information:
            # A test cannot fail to generate without an explicit engineering reason
            raise ValueError("insufficient_information requires an explanation")

        # Normalize the internal state for null scenarios cleanly
        if self.status == "insufficient_information":
            self.steps = []
            self.pass_criteria = self.fail_criteria = None

        return self