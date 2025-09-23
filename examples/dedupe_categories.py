"""Utility script to merge duplicate categories created by earlier demo runs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from memoir_ai import Category, Chunk


def iter_duplicate_groups(session: Session) -> Iterable[tuple[int | None, str, int]]:
    """Yield (parent_id, name, count) for categories with duplicates."""
    return (
        session.query(
            Category.parent_id,
            Category.name,
            func.count(Category.id).label("duplicate_count"),
        )
        .group_by(Category.parent_id, Category.name)
        .having(func.count(Category.id) > 1)
        .all()
    )


def merge_duplicate_categories(session: Session) -> int:
    """Merge categories that share the same parent and name.

    Returns the number of category rows that were removed.
    """

    removed = 0
    while True:
        duplicate_groups = list(iter_duplicate_groups(session))
        if not duplicate_groups:
            break

        for parent_id, name, _ in duplicate_groups:
            # Order deterministically so we retain the oldest entry.
            categories = (
                session.query(Category)
                .filter_by(parent_id=parent_id, name=name)
                .order_by(Category.id)
                .all()
            )
            if not categories:
                continue

            keeper = categories[0]
            for duplicate in categories[1:]:
                session.query(Category).filter_by(parent_id=duplicate.id).update(
                    {Category.parent_id: keeper.id}, synchronize_session=False
                )
                session.query(Chunk).filter_by(category_id=duplicate.id).update(
                    {Chunk.category_id: keeper.id}, synchronize_session=False
                )
                session.delete(duplicate)
                removed += 1

        session.flush()

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default="sqlite:///example_database.db",
        help="SQLAlchemy database URL to clean (default: sqlite:///example_database.db)",
    )
    args = parser.parse_args()

    engine = create_engine(args.database_url)
    with Session(engine) as session:
        print("🔍 Scanning for duplicate categories...")
        removed = merge_duplicate_categories(session)
        session.commit()

    if removed:
        print(f"✅ Removed {removed} duplicate category rows.")
    else:
        print("✅ No duplicate categories found.")


if __name__ == "__main__":
    main()
