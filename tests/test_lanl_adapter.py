"""Unit tests for LANL adapter (Module M3.1).

Tests iter_lanl_events() and build_lanl_graph() using tiny temporary .gz
fixtures. Does not use the real 7.6GB auth.txt.gz file.
"""

import gzip
import os
import tempfile
import time
from pathlib import Path

import pytest

from ml.preprocessing.lanl_adapter import (
    LANLAdapterError,
    LANLParseError,
    build_lanl_graph,
    iter_lanl_events,
)
from ml.preprocessing.schema import CanonicalEvent, EVENT_TYPE_AUTHENTICATION


def create_temp_lanl_gz(lines: list[str]) -> Path:
    """Create a temporary .gz file with the given lines."""
    fd, path = tempfile.mkstemp(suffix=".txt.gz")
    os.close(fd)  # Close the file descriptor
    try:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        return Path(path)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def cleanup_temp(path: Path) -> None:
    """Clean up temp file with delay for Windows file locking."""
    time.sleep(0.05)
    path.unlink(missing_ok=True)


class TestIterLANLEvents:
    """Test iter_lanl_events() streaming parser."""

    def test_valid_success_record(self):
        """Test parsing a valid successful authentication record."""
        lines = [
            "1,ANONYMOUS LOGON@C586,ANONYMOUS LOGON@C586,C1250,C586,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            events = list(iter_lanl_events(path))
            assert len(events) == 1

            event = events[0]
            assert isinstance(event, CanonicalEvent)
            assert event.timestamp == 1.0
            assert event.user == "ANONYMOUS LOGON@C586"
            assert event.source_host == "C1250"
            assert event.destination_host == "C586"
            assert event.event_type == EVENT_TYPE_AUTHENTICATION
            assert event.success is True
            assert event.event_id.startswith("lanl_")
        finally:
            cleanup_temp(path)

    def test_valid_fail_record(self):
        """Test parsing a valid failed authentication record."""
        lines = [
            "100,U620@DOM1,U620@DOM1,C100,C200,Kerberos,Network,LogOn,Fail"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            events = list(iter_lanl_events(path))
            assert len(events) == 1

            event = events[0]
            assert event.timestamp == 100.0
            assert event.success is False
            assert event.user == "U620@DOM1"
        finally:
            cleanup_temp(path)

    def test_multiple_records(self):
        """Test parsing multiple records."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success",
            "2,USER2@DOM1,USER2@DOM1,C3,C4,Kerberos,Service,LogOff,Success",
            "3,USER3@DOM1,USER3@DOM1,C5,C6,Negotiate,Batch,TGS,Fail",
        ]
        path = create_temp_lanl_gz(lines)
        try:
            events = list(iter_lanl_events(path))
            assert len(events) == 3
            assert events[0].timestamp == 1.0
            assert events[1].timestamp == 2.0
            assert events[2].timestamp == 3.0
        finally:
            cleanup_temp(path)

    def test_deterministic_event_id(self):
        """Test that event_id is deterministic for the same record."""
        lines = [
            "1,USER@DOM1,USER@DOM1,C1,C2,NTLM,Network,LogOn,Success"
        ]
        path1 = create_temp_lanl_gz(lines)
        path2 = create_temp_lanl_gz(lines)
        try:
            events1 = list(iter_lanl_events(path1))
            events2 = list(iter_lanl_events(path2))
            # Same input line should produce same event_id
            assert events1[0].event_id == events2[0].event_id
        finally:
            cleanup_temp(path1)
            cleanup_temp(path2)

    def test_limit_parameter(self):
        """Test that limit parameter restricts output."""
        lines = [
            f"{i},USER{i}@DOM1,USER{i}@DOM1,C{i},C{i+1},NTLM,Network,LogOn,Success"
            for i in range(10)
        ]
        path = create_temp_lanl_gz(lines)
        try:
            events = list(iter_lanl_events(path, limit=5))
            assert len(events) == 5
        finally:
            cleanup_temp(path)

    def test_skip_empty_lines(self):
        """Test that empty lines are skipped without error."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success",
            "",
            "2,USER2@DOM1,USER2@DOM1,C3,C4,Kerberos,Network,LogOff,Success",
            "",
            "",
        ]
        path = create_temp_lanl_gz(lines)
        try:
            events = list(iter_lanl_events(path))
            assert len(events) == 2
        finally:
            cleanup_temp(path)

    def test_malformed_column_count(self):
        """Test that wrong column count raises LANLParseError."""
        lines = [
            "1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success"  # 8 columns instead of 9
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError, match="expected 9 columns"):
                list(iter_lanl_events(path))
        finally:
            cleanup_temp(path)

    def test_malformed_empty_timestamp(self):
        """Test that empty timestamp raises LANLParseError."""
        lines = [
            ",USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError, match="timestamp is empty"):
                list(iter_lanl_events(path))
        finally:
            cleanup_temp(path)

    def test_malformed_non_numeric_timestamp(self):
        """Test that non-numeric timestamp raises LANLParseError."""
        lines = [
            "abc,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError, match="not a valid number"):
                list(iter_lanl_events(path))
        finally:
            cleanup_temp(path)

    def test_malformed_empty_user(self):
        """Test that empty user raises LANLParseError."""
        lines = [
            "1,,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError, match="user is empty"):
                list(iter_lanl_events(path))
        finally:
            cleanup_temp(path)

    def test_malformed_empty_source_host(self):
        """Test that empty source_host raises LANLParseError."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,,C2,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError, match="source_host is empty"):
                list(iter_lanl_events(path))
        finally:
            cleanup_temp(path)

    def test_malformed_empty_dest_host(self):
        """Test that empty destination_host raises LANLParseError."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError, match="destination_host is empty"):
                list(iter_lanl_events(path))
        finally:
            cleanup_temp(path)

    def test_malformed_invalid_status(self):
        """Test that invalid status value raises LANLParseError."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Unknown"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError, match="not 'Success' or 'Fail'"):
                list(iter_lanl_events(path))
        finally:
            cleanup_temp(path)

    def test_missing_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            list(iter_lanl_events(Path("/nonexistent/file.gz")))

    def test_streaming_iterator(self):
        """Test that events are yielded incrementally (not loaded all at once)."""
        lines = [f"{i},USER{i}@DOM1,USER{i}@DOM1,C{i},C{i+1},NTLM,Network,LogOn,Success" for i in range(1000)]
        path = create_temp_lanl_gz(lines)
        try:
            # Should return a generator, not a list
            gen = iter_lanl_events(path)
            assert hasattr(gen, "__iter__")
            assert hasattr(gen, "__next__")
            # Consume first event
            event = next(gen)
            assert event.timestamp == 0.0
            # Consume rest to close the generator/file
            list(gen)
        finally:
            cleanup_temp(path)

    def test_timestamp_float_conversion(self):
        """Test that timestamps are converted to float."""
        lines = [
            "123456789,USER@DOM1,USER@DOM1,C1,C2,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            events = list(iter_lanl_events(path))
            assert events[0].timestamp == 123456789.0
            assert isinstance(events[0].timestamp, float)
        finally:
            cleanup_temp(path)

    def test_status_case_insensitive(self):
        """Test that status parsing is case-insensitive."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,SUCCESS",
            "2,USER2@DOM1,USER2@DOM1,C3,C4,NTLM,Network,LogOn,success",
            "3,USER3@DOM1,USER3@DOM1,C5,C6,NTLM,Network,LogOn,FAIL",
            "4,USER4@DOM1,USER4@DOM1,C7,C8,NTLM,Network,LogOn,fail",
        ]
        path = create_temp_lanl_gz(lines)
        try:
            events = list(iter_lanl_events(path))
            assert events[0].success is True
            assert events[1].success is True
            assert events[2].success is False
            assert events[3].success is False
        finally:
            cleanup_temp(path)


class TestBuildLANLGraph:
    """Test build_lanl_graph() integration."""

    def test_build_empty_graph(self):
        """Test building a graph from a file with no valid records."""
        lines = [""]
        path = create_temp_lanl_gz(lines)
        try:
            graph = build_lanl_graph(path)
            assert graph.number_of_events() == 0
            assert graph.number_of_nodes() == 0
        finally:
            cleanup_temp(path)

    def test_build_single_event_graph(self):
        """Test building a graph with a single event."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success"
        ]
        path = create_temp_lanl_gz(lines)
        try:
            graph = build_lanl_graph(path)
            assert graph.number_of_events() == 1
            assert graph.number_of_nodes() == 3  # user node + 2 host nodes
        finally:
            cleanup_temp(path)

    def test_build_multi_event_graph(self):
        """Test building a graph with multiple events."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success",
            "2,USER2@DOM1,USER2@DOM1,C3,C4,Kerberos,Network,LogOn,Success",
            "3,USER1@DOM1,USER1@DOM1,C5,C6,NTLM,Service,LogOff,Success",
        ]
        path = create_temp_lanl_gz(lines)
        try:
            graph = build_lanl_graph(path)
            assert graph.number_of_events() == 3
            # Should have at least 2 unique users and 4 unique hosts
            assert graph.number_of_nodes() >= 6
        finally:
            cleanup_temp(path)

    def test_build_with_limit(self):
        """Test building a graph with a limit."""
        lines = [f"{i},USER{i}@DOM1,USER{i}@DOM1,C{i},C{i+1},NTLM,Network,LogOn,Success" for i in range(100)]
        path = create_temp_lanl_gz(lines)
        try:
            graph = build_lanl_graph(path, limit=10)
            assert graph.number_of_events() == 10
        finally:
            cleanup_temp(path)

    def test_build_malformed_raises_error(self):
        """Test that malformed records raise error during graph build."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success",
            "MALFORMED",  # Invalid record
        ]
        path = create_temp_lanl_gz(lines)
        try:
            with pytest.raises(LANLParseError):
                build_lanl_graph(path)
        finally:
            cleanup_temp(path)

    def test_graph_edge_structure(self):
        """Test that graph edges have correct structure."""
        lines = [
            "1,USER1@DOM1,USER1@DOM1,C1,C2,NTLM,Network,LogOn,Success",
            "2,USER1@DOM1,USER1@DOM1,C1,C3,NTLM,Network,LogOn,Success",
        ]
        path = create_temp_lanl_gz(lines)
        try:
            graph = build_lanl_graph(path)
            # Check that we have edges
            edges = list(graph.iter_edges())
            assert len(edges) > 0
            # We should have 2 edges per event: user→host (identity) and host→host (movement)
            assert len(edges) == 4
            
            # Classify edges by type
            identity_edges = [e for e in edges if e[0].startswith("user:")]
            movement_edges = [e for e in edges if e[0].startswith("host:")]
            
            # Should have equal numbers of identity and movement edges
            assert len(identity_edges) == 2
            assert len(movement_edges) == 2
            
            # All edges should go to hosts
            for src, dst, key, attrs in edges:
                assert dst.startswith("host:")
        finally:
            cleanup_temp(path)
