"""
Integration tests for database functionality.
"""

import os
import tempfile

import pytest

from memoir_ai.config import MemoirAIConfig
from memoir_ai.database.engine import DatabaseManager
from memoir_ai.database.models import Category, CategoryLimits, Chunk, ContextualHelper
from memoir_ai.exceptions import DatabaseError


class TestDatabaseIntegration:
    """Test complete database integration."""

    def test_complete_workflow(self) -> None:
        """Test complete database workflow with all models."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)
        db_manager.create_tables()

        with db_manager.get_session() as session:
            # Create category hierarchy
            tech_category = Category(name="Technology", level=1, parent_id=None)
            session.add(tech_category)
            session.flush()  # Get the ID

            ai_category = Category(name="AI/ML", level=2, parent_id=tech_category.id)
            session.add(ai_category)
            session.flush()

            nlp_category = Category(name="NLP", level=3, parent_id=ai_category.id)
            session.add(nlp_category)
            session.flush()

            # Create contextual helper
            helper = ContextualHelper(
                source_id="ai_paper_2024",
                helper_text="This paper discusses advances in natural language processing and machine learning.",
                token_count=15,
                is_user_provided=False,
            )
            session.add(helper)

            # Create chunks
            chunk1 = Chunk(
                content="Natural language processing has made significant advances in recent years.",
                token_count=12,
                category_id=nlp_category.id,
                source_id="ai_paper_2024",
            )

            chunk2 = Chunk(
                content="Machine learning models are becoming more sophisticated and capable.",
                token_count=10,
                category_id=ai_category.id,
                source_id="ai_paper_2024",
            )

            session.add_all([chunk1, chunk2])

            # Create category limits
            limits = CategoryLimits(level=1, max_categories=100)
            session.add(limits)

        # Verify data was saved correctly
        with db_manager.get_session() as session:
            # Test category hierarchy
            categories = session.query(Category).all()
            assert len(categories) == 3

            tech = session.query(Category).filter_by(name="Technology").first()
            assert tech.level == 1
            assert tech.parent_id is None
            assert len(tech.children) == 1

            ai = session.query(Category).filter_by(name="AI/ML").first()
            assert ai.level == 2
            assert ai.parent_id == tech.id
            assert ai.parent == tech

            nlp = session.query(Category).filter_by(name="NLP").first()
            assert nlp.level == 3
            assert nlp.parent_id == ai.id

            # Test path methods
            assert nlp.get_full_path() == ["Technology", "AI/ML", "NLP"]
            assert nlp.get_path_string() == "Technology → AI/ML → NLP"

            # Test chunks
            chunks = session.query(Chunk).all()
            assert len(chunks) == 2

            nlp_chunks = session.query(Chunk).filter_by(category_id=nlp.id).all()
            assert len(nlp_chunks) == 1
            assert nlp_chunks[0].content.startswith("Natural language processing")

            # Test contextual helper
            helpers = session.query(ContextualHelper).all()
            assert len(helpers) == 1
            assert helpers[0].source_id == "ai_paper_2024"
            assert helpers[0].token_count == 15

            # Test category limits
            limits = session.query(CategoryLimits).all()
            assert len(limits) == 1
            assert limits[0].level == 1
            assert limits[0].max_categories == 100

        # Test table info
        info = db_manager.get_table_info()
        assert info["categories"]["row_count"] == 3
        assert info["chunks"]["row_count"] == 2
        assert info["contextual_helpers"]["row_count"] == 1
        assert info["category_limits"]["row_count"] == 1

        db_manager.close()

    def test_constraint_violations(self) -> None:
        """Test that database constraints are properly enforced."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)
        db_manager.create_tables()

        with db_manager.get_session() as session:
            # Create parent category
            parent = Category(name="Technology", level=1, parent_id=None)
            session.add(parent)
            session.flush()

            # Create first child
            child1 = Category(name="AI", level=2, parent_id=parent.id)
            session.add(child1)
            session.commit()

        # Test unique name constraint violation
        with pytest.raises(DatabaseError):
            with db_manager.get_session() as session:
                # Try to create another child with same name
                child2 = Category(name="AI", level=2, parent_id=parent.id)
                session.add(child2)
                # This should fail due to unique constraint

        # Test contextual helper unique source_id constraint
        with db_manager.get_session() as session:
            helper1 = ContextualHelper(
                source_id="test_source", helper_text="First helper", token_count=2
            )
            session.add(helper1)

        with pytest.raises(DatabaseError):
            with db_manager.get_session() as session:
                helper2 = ContextualHelper(
                    source_id="test_source",  # Same source_id
                    helper_text="Second helper",
                    token_count=2,
                )
                session.add(helper2)
                # This should fail due to unique constraint

        db_manager.close()

    def test_file_database(self) -> None:
        """Test with file-based SQLite database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            config = MemoirAIConfig(database_url=f"sqlite:///{db_path}")
            db_manager = DatabaseManager(config)
            db_manager.create_tables()

            # Add some data
            with db_manager.get_session() as session:
                category = Category(name="Test Category", level=1, parent_id=None)
                session.add(category)

            db_manager.close()

            # Reopen database and verify data persists
            db_manager2 = DatabaseManager(config)
            with db_manager2.get_session() as session:
                categories = session.query(Category).all()
                assert len(categories) == 1
                assert categories[0].name == "Test Category"

            db_manager2.close()

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
