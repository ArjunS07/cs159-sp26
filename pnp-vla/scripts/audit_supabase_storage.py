#!/usr/bin/env python3
"""Read-only inventory of a Supabase Storage bucket.

The Storage API is folder-oriented, so this walks prefixes concurrently.  It
downloads object metadata only; it never downloads object bodies or mutates
Storage/Postgres.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import httpx


@dataclass(frozen=True)
class StoredObject:
    path: str
    size: int
    created_at: str | None
    updated_at: str | None
    mimetype: str | None


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def _list_directory(
    client: httpx.AsyncClient, endpoint: str, bucket: str, prefix: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = await client.post(
            f"{endpoint}/storage/v1/object/list/{bucket}",
            json={
                "prefix": prefix,
                "limit": 1000,
                "offset": offset,
                "sortBy": {"column": "name", "order": "asc"},
            },
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < 1000:
            return rows
        offset += len(page)


async def inventory(
    endpoint: str, service_key: str, bucket: str, concurrency: int
) -> list[StoredObject]:
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": "application/json",
    }
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put("")
    seen = {""}
    objects: list[StoredObject] = []
    lock = asyncio.Lock()
    directory_count = 0

    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        async def worker() -> None:
            nonlocal directory_count
            while True:
                prefix = await queue.get()
                if prefix is None:
                    queue.task_done()
                    return
                try:
                    rows = await _list_directory(client, endpoint, bucket, prefix)
                    discovered: list[str] = []
                    found: list[StoredObject] = []
                    for row in rows:
                        name = row.get("name", "")
                        path = f"{prefix}/{name}".strip("/")
                        if row.get("id") is None:
                            discovered.append(path)
                            continue
                        metadata = row.get("metadata") or {}
                        found.append(
                            StoredObject(
                                path=path,
                                size=int(metadata.get("size") or 0),
                                created_at=row.get("created_at"),
                                updated_at=row.get("updated_at"),
                                mimetype=metadata.get("mimetype"),
                            )
                        )
                    async with lock:
                        directory_count += 1
                        objects.extend(found)
                        for child in discovered:
                            if child not in seen:
                                seen.add(child)
                                await queue.put(child)
                        if directory_count % 100 == 0:
                            size = sum(item.size for item in objects)
                            print(
                                f"walked {directory_count:,} folders; "
                                f"found {len(objects):,} objects / {format_bytes(size)}",
                                file=sys.stderr,
                                flush=True,
                            )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
    return objects


def format_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def print_profile(objects: list[StoredObject], depth: int, limit: int) -> None:
    aggregate: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in objects:
        prefix = "/".join(item.path.split("/")[:depth])
        aggregate[prefix][0] += item.size
        aggregate[prefix][1] += 1
    print(f"\nStorage by path prefix (depth={depth})")
    for prefix, (size, count) in sorted(
        aggregate.items(), key=lambda pair: pair[1][0], reverse=True
    )[:limit]:
        print(f"{format_bytes(size):>12}  {count:>8,}  {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--bucket", default="artifacts")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    _load_env(args.env)
    endpoint = os.environ["SUPABASE_URL"].rstrip("/")
    service_key = os.environ["SUPABASE_SERVICE_KEY"]
    objects = asyncio.run(
        inventory(endpoint, service_key, args.bucket, args.concurrency)
    )
    objects.sort(key=lambda item: item.path)
    total = sum(item.size for item in objects)
    print(f"\nTotal: {len(objects):,} objects / {format_bytes(total)}")
    for depth in (1, 2, 3):
        print_profile(objects, depth, args.limit)
    print("\nLargest objects")
    for item in sorted(objects, key=lambda item: item.size, reverse=True)[:30]:
        print(f"{format_bytes(item.size):>12}  {item.created_at or '':26}  {item.path}")
    if args.output:
        args.output.write_text(json.dumps([asdict(item) for item in objects]))
        print(f"\nWrote metadata inventory to {args.output}")


if __name__ == "__main__":
    main()
