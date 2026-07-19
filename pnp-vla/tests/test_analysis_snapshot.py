from types import SimpleNamespace

from analysis.snapshot import paginated_rows


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
