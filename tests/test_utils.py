import pytest
from utils import TTLFilter

class TestTTLFilter:
    """Test cases for the TTLFilter class."""

    def test_cache_set_and_get(self):
        """Test setting and getting values from cache."""
        cache = TTLFilter()
        cache.add_item("key1", "value1", 10)
        result = cache.get_item("key1")
        assert result == "value1"

    def test_cache_expiration(self):
        """Test that cache items expire correctly."""
        cache = TTLFilter()
        cache.add_item("key1", "value1", -1)  # Already expired
        result = cache.get_item("key1")
        assert result is None

    def test_cache_cleanup(self):
        """Test cleaning up expired items."""
        cache = TTLFilter()
        cache.add_item("key1", "value1", -1)   # Expired
        cache.add_item("key2", "value2", 10)   # Not expired
        expired_count = cache.cleanup_expired()
        assert expired_count == 1
        assert cache.get_item("key1") is None
        assert cache.get_item("key2") == "value2"