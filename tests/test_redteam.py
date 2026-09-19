"""Unit tests for the M3.2 redteam ground truth module.

All objects are tiny deterministic in-memory software fixtures; they
are not research data and do not depend on any real dataset.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from ml.evaluation.redteam import (
    RedteamError,
    RedteamGroundTruth,
    RedteamParseError,
    RedteamRecord,
    iter_redteam_events,
    matches_canonical_event,
)
from ml.preprocessing.schema import CanonicalEvent


def create_temp_redteam_gz(lines):
    """Create temporary redteam .gz file with given lines (no header)."""
    import gzip

    fd, path = tempfile.mkstemp(suffix=".gz")
    os.close(fd)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def cleanup_temp(path):
    """Clean up temporary file after test."""
    try:
        Path(path).unlink()
        time.sleep(0.05)  # Allow Windows file locking to clear
    except FileNotFoundError:
        pass


# ============================================================================
# Parsing Tests
# ============================================================================


class TestIterRedteamEvents:
    """Tests for iter_redteam_events() function."""

    def test_valid_record_parsing(self):
        """Test that a valid record is parsed correctly."""
        lines = ["150885,U620@DOM1,C17693,C1003"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            assert len(records) == 1
            record = records[0]
            assert record.timestamp == 150885.0
            assert record.user == "U620@DOM1"
            assert record.source_host == "C17693"
            assert record.destination_host == "C1003"
        finally:
            cleanup_temp(path)

    def test_all_four_fields_preserved(self):
        """Test that all four fields are stored in RedteamRecord."""
        lines = ["123,UserX,HostA,HostB"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            record = records[0]
            assert hasattr(record, "redteam_id")
            assert hasattr(record, "timestamp")
            assert hasattr(record, "user")
            assert hasattr(record, "source_host")
            assert hasattr(record, "destination_host")
        finally:
            cleanup_temp(path)

    def test_deterministic_redteam_id(self):
        """Test that redteam_ids are deterministic across runs."""
        lines = ["100,User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            records1 = list(iter_redteam_events(path))
            records2 = list(iter_redteam_events(path))
            assert records1[0].redteam_id == records2[0].redteam_id
            assert records1[0].redteam_id.startswith("redteam_")
        finally:
            cleanup_temp(path)

    def test_multiple_records(self):
        """Test parsing multiple records."""
        lines = [
            "100,User1,Host1,Host2",
            "200,User2,Host2,Host3",
            "300,User3,Host3,Host1",
        ]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            assert len(records) == 3
            assert [r.timestamp for r in records] == [100.0, 200.0, 300.0]
        finally:
            cleanup_temp(path)

    def test_duplicate_records_preserved(self):
        """Test that duplicate records are not silently deleted."""
        lines = [
            "100,User1,Host1,Host2",
            "100,User1,Host1,Host2",  # exact duplicate
            "100,User1,Host1,Host2",  # another duplicate
        ]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            assert len(records) == 3
            # All have same (timestamp, user, source, dest) but different line numbers
            # so redteam_ids should be different
            ids = {r.redteam_id for r in records}
            assert len(ids) == 3
        finally:
            cleanup_temp(path)

    def test_empty_lines_skipped(self):
        """Test that empty lines are skipped without error."""
        lines = [
            "100,User1,Host1,Host2",
            "",  # empty line
            "200,User2,Host2,Host3",
            "",  # another empty line
        ]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            assert len(records) == 2
        finally:
            cleanup_temp(path)

    def test_wrong_column_count_rejected(self):
        """Test that records with wrong column count are rejected."""
        lines = ["100,User1,Host1"]  # only 3 columns
        path = create_temp_redteam_gz(lines)
        try:
            with pytest.raises(RedteamParseError, match="4 columns"):
                list(iter_redteam_events(path))
        finally:
            cleanup_temp(path)

    def test_extra_columns_rejected(self):
        """Test that records with extra columns are rejected."""
        lines = ["100,User1,Host1,Host2,Extra"]  # 5 columns
        path = create_temp_redteam_gz(lines)
        try:
            with pytest.raises(RedteamParseError, match="4 columns"):
                list(iter_redteam_events(path))
        finally:
            cleanup_temp(path)

    def test_empty_timestamp_rejected(self):
        """Test that empty timestamp is rejected."""
        lines = [",User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            with pytest.raises(RedteamParseError, match="timestamp is empty"):
                list(iter_redteam_events(path))
        finally:
            cleanup_temp(path)

    def test_empty_user_rejected(self):
        """Test that empty user is rejected."""
        lines = ["100,,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            with pytest.raises(RedteamParseError, match="user is empty"):
                list(iter_redteam_events(path))
        finally:
            cleanup_temp(path)

    def test_empty_source_host_rejected(self):
        """Test that empty source_host is rejected."""
        lines = ["100,User1,,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            with pytest.raises(RedteamParseError, match="source_host is empty"):
                list(iter_redteam_events(path))
        finally:
            cleanup_temp(path)

    def test_empty_destination_host_rejected(self):
        """Test that empty destination_host is rejected."""
        lines = ["100,User1,Host1,"]
        path = create_temp_redteam_gz(lines)
        try:
            with pytest.raises(RedteamParseError, match="destination_host is empty"):
                list(iter_redteam_events(path))
        finally:
            cleanup_temp(path)

    def test_non_numeric_timestamp_rejected(self):
        """Test that non-numeric timestamp is rejected."""
        lines = ["abc,User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            with pytest.raises(RedteamParseError, match="not a valid number"):
                list(iter_redteam_events(path))
        finally:
            cleanup_temp(path)

    def test_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list(iter_redteam_events("/nonexistent/path.gz"))

    def test_streaming_iterator_behavior(self):
        """Test that iterator doesn't load entire file upfront."""
        lines = ["100,User1,Host1,Host2", "200,User2,Host2,Host3"]
        path = create_temp_redteam_gz(lines)
        try:
            gen = iter_redteam_events(path)
            # Get first record
            record1 = next(gen)
            assert record1.timestamp == 100.0
            # Get second record
            record2 = next(gen)
            assert record2.timestamp == 200.0
            # Consuming rest to close file
            list(gen)
        finally:
            cleanup_temp(path)


# ============================================================================
# Ground Truth Tests
# ============================================================================


class TestRedteamGroundTruth:
    """Tests for RedteamGroundTruth class."""

    def test_load_from_iterator(self):
        """Test loading records from iterator."""
        lines = [
            "100,User1,Host1,Host2",
            "200,User2,Host2,Host3",
        ]
        path = create_temp_redteam_gz(lines)
        try:
            records = iter_redteam_events(path)
            gt = RedteamGroundTruth(records)
            assert gt.number_of_records() == 2
        finally:
            cleanup_temp(path)

    def test_load_from_file(self):
        """Test from_file() class method."""
        lines = ["100,User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            assert gt.number_of_records() == 1
        finally:
            cleanup_temp(path)

    def test_get_record_by_id(self):
        """Test lookup of record by redteam_id."""
        lines = ["100,User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            records = list(gt.iter_records())
            record_id = records[0].redteam_id
            retrieved = gt.get_record(record_id)
            assert retrieved is not None
            assert retrieved.user == "User1"
        finally:
            cleanup_temp(path)

    def test_get_nonexistent_record_returns_none(self):
        """Test that looking up non-existent record returns None."""
        lines = ["100,User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            result = gt.get_record("nonexistent")
            assert result is None
        finally:
            cleanup_temp(path)

    def test_iter_records_sorted_by_timestamp(self):
        """Test that iter_records returns chronologically sorted records."""
        lines = [
            "300,User3,Host3,Host1",
            "100,User1,Host1,Host2",
            "200,User2,Host2,Host3",
        ]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            records = list(gt.iter_records())
            timestamps = [r.timestamp for r in records]
            assert timestamps == [100.0, 200.0, 300.0]
        finally:
            cleanup_temp(path)

    def test_filter_by_timestamp(self):
        """Test filtering records by timestamp window."""
        lines = [
            "100,User1,Host1,Host2",
            "200,User2,Host2,Host3",
            "300,User3,Host3,Host1",
        ]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            results = gt.filter_by_timestamp(100.0, 200.0)
            assert len(results) == 2
            assert all(100.0 <= r.timestamp <= 200.0 for r in results)
        finally:
            cleanup_temp(path)

    def test_filter_by_user(self):
        """Test filtering records by user."""
        lines = [
            "100,User1,Host1,Host2",
            "200,User1,Host2,Host3",
            "300,User2,Host3,Host1",
        ]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            results = gt.filter_by_user("User1")
            assert len(results) == 2
            assert all(r.user == "User1" for r in results)
        finally:
            cleanup_temp(path)

    def test_filter_by_source_host(self):
        """Test filtering records by source host."""
        lines = [
            "100,User1,Host1,Host2",
            "200,User2,Host1,Host3",
            "300,User3,Host2,Host1",
        ]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            results = gt.filter_by_source_host("Host1")
            assert len(results) == 2
            assert all(r.source_host == "Host1" for r in results)
        finally:
            cleanup_temp(path)

    def test_filter_by_destination_host(self):
        """Test filtering records by destination host."""
        lines = [
            "100,User1,Host1,Host2",
            "200,User2,Host3,Host2",
            "300,User3,Host2,Host1",
        ]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            results = gt.filter_by_destination_host("Host2")
            assert len(results) == 2
            assert all(r.destination_host == "Host2" for r in results)
        finally:
            cleanup_temp(path)

    def test_filter_by_nonexistent_user_returns_empty(self):
        """Test that filtering by non-existent user returns empty list."""
        lines = ["100,User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            gt = RedteamGroundTruth.from_file(path)
            results = gt.filter_by_user("NonExistentUser")
            assert results == []
        finally:
            cleanup_temp(path)


# ============================================================================
# Event Matching Tests
# ============================================================================


class TestMatchesCanonicalEvent:
    """Tests for matches_canonical_event() function."""

    def test_exact_match(self):
        """Test that exact matching works."""
        lines = ["150885,U620@DOM1,C17693,C1003"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            record = records[0]
            event = CanonicalEvent(
                event_id="test1",
                timestamp=150885.0,
                user="U620@DOM1",
                source_host="C17693",
                destination_host="C1003",
                event_type="authentication",
                success=True,
            )
            assert matches_canonical_event(record, event)
        finally:
            cleanup_temp(path)

    def test_no_match_different_timestamp(self):
        """Test that different timestamp means no match."""
        lines = ["150885,U620@DOM1,C17693,C1003"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            record = records[0]
            event = CanonicalEvent(
                event_id="test1",
                timestamp=150886.0,  # different
                user="U620@DOM1",
                source_host="C17693",
                destination_host="C1003",
                event_type="authentication",
                success=True,
            )
            assert not matches_canonical_event(record, event)
        finally:
            cleanup_temp(path)

    def test_no_match_different_user(self):
        """Test that different user means no match."""
        lines = ["150885,U620@DOM1,C17693,C1003"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            record = records[0]
            event = CanonicalEvent(
                event_id="test1",
                timestamp=150885.0,
                user="U621@DOM1",  # different
                source_host="C17693",
                destination_host="C1003",
                event_type="authentication",
                success=True,
            )
            assert not matches_canonical_event(record, event)
        finally:
            cleanup_temp(path)

    def test_no_match_different_source_host(self):
        """Test that different source_host means no match."""
        lines = ["150885,U620@DOM1,C17693,C1003"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            record = records[0]
            event = CanonicalEvent(
                event_id="test1",
                timestamp=150885.0,
                user="U620@DOM1",
                source_host="C17694",  # different
                destination_host="C1003",
                event_type="authentication",
                success=True,
            )
            assert not matches_canonical_event(record, event)
        finally:
            cleanup_temp(path)

    def test_no_match_different_destination_host(self):
        """Test that different destination_host means no match."""
        lines = ["150885,U620@DOM1,C17693,C1003"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            record = records[0]
            event = CanonicalEvent(
                event_id="test1",
                timestamp=150885.0,
                user="U620@DOM1",
                source_host="C17693",
                destination_host="C1004",  # different
                event_type="authentication",
                success=True,
            )
            assert not matches_canonical_event(record, event)
        finally:
            cleanup_temp(path)

    def test_match_ignores_event_id_and_success(self):
        """Test that event_id and success don't affect matching."""
        lines = ["100,User1,Host1,Host2"]
        path = create_temp_redteam_gz(lines)
        try:
            records = list(iter_redteam_events(path))
            record = records[0]
            # Same (timestamp, user, source, dest) but different event_id and success
            event = CanonicalEvent(
                event_id="different_id",
                timestamp=100.0,
                user="User1",
                source_host="Host1",
                destination_host="Host2",
                event_type="authentication",
                success=False,  # different success value
            )
            assert matches_canonical_event(record, event)
        finally:
            cleanup_temp(path)


# ============================================================================
# Integration Tests
# ============================================================================


def test_raw_file_not_modified():
    """Test that raw redteam.txt.gz is never modified."""
    path = Path("data/raw/lanl/redteam.txt.gz")
    if not path.exists():
        pytest.skip("redteam.txt.gz not found")

    # Get original stat
    stat_before = path.stat()
    mtime_before = stat_before.st_mtime

    # Parse records
    gt = RedteamGroundTruth.from_file(path)
    assert gt.number_of_records() > 0

    # Verify file not modified
    stat_after = path.stat()
    mtime_after = stat_after.st_mtime
    assert mtime_before == mtime_after
    assert stat_before.st_size == stat_after.st_size
