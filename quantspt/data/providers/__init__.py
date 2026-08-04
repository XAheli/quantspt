"""Pluggable data providers following the TET pattern.

Each provider implements ``transform_query() → extract_data() → transform_data()``
and is discoverable via entry points or runtime registration.
"""

__all__: list[str] = []
