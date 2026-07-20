from types import SimpleNamespace

import pandas as pd

from analysis.snapshot import paginated_rows, verify_artifact_references


class Query:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_): return self
    def eq(self, *_): return self
    def range(self, start, end):
        self.page = self.rows[start:end + 1]
        return self
    def execute(self): return SimpleNamespace(data=self.page)


class Client:
    def __init__(self, rows): self.rows = rows
    def table(self, _): return Query(self.rows)


def test_paginated_extraction_reads_final_partial_page():
    rows = [{"id": i} for i in range(7)]
    assert paginated_rows(Client(rows), "rollouts", page_size=3) == rows


def test_artifact_verification_lists_folders_and_reports_missing():
    class Bucket:
        def list(self, folder, options):
            assert folder == "pcp_chunks"
            return [{"name": "present.parquet"}]

    class Storage:
        def from_(self, bucket):
            assert bucket == "artifacts"
            return Bucket()

    client = SimpleNamespace(storage=Storage())
    frame = pd.DataFrame({"pcp_chunks_path": [
        "pcp_chunks/present.parquet", "pcp_chunks/missing.parquet"]})
    result = verify_artifact_references(client, frame)["pcp_chunks_path"]
    assert result["referenced"] == 2
    assert result["verified"] == 1
    assert result["status"] == "invalid"
    assert result["missing"] == ["pcp_chunks/missing.parquet"]
