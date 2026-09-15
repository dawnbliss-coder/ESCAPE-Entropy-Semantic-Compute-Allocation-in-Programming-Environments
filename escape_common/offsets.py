"""Byte <-> character offset utilities for the frozen UTF-8 corpus.

Every ESCAPE offset is a BYTE offset into corpus/{domain}/{file_id}.bin
(docs/SCHEMA.md). A tool that only produces character (code point) offsets, such
as a constituency parser run on decoded text, must be mapped through these
functions, and the mapping must be checked before its output is trusted.

Conventions:
  - maps have length len + 1, so an exclusive end offset (one past the last
    character or byte) converts too;
  - a byte offset inside a multi-byte UTF-8 sequence has no character offset:
    byte_to_char_map marks it -1 and bytes_to_chars raises.
"""

from __future__ import annotations

import numpy as np


def utf8_roundtrip_ok(content: bytes) -> bool:
    """True iff `content` is valid UTF-8 and decoding then re-encoding it gives
    back exactly the same bytes (strict decoder, no replacement characters)."""
    content = bytes(content)
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return text.encode("utf-8") == content


def char_start_mask(content: bytes) -> np.ndarray:
    """mask[j] is True iff byte j starts a UTF-8 character, i.e. it is not a
    continuation byte (0b10xxxxxx). Only meaningful for valid UTF-8."""
    b = np.frombuffer(bytes(content), dtype=np.uint8)
    return (b & 0xC0) != 0x80


def char_to_byte_map(text: str) -> np.ndarray:
    """m[i] = byte offset of character i in text.encode('utf-8');
    m[len(text)] = total number of bytes. int64."""
    content = text.encode("utf-8")  # raises UnicodeEncodeError on lone surrogates
    starts = np.flatnonzero(char_start_mask(content))
    if starts.size != len(text):  # cannot happen for a str that encodes
        raise ValueError("character count mismatch while building the offset map")
    out = np.empty(len(text) + 1, dtype=np.int64)
    out[:-1] = starts
    out[-1] = len(content)
    return out


def byte_to_char_map(content: bytes) -> np.ndarray:
    """m[j] = index of the character that starts at byte j, or -1 if byte j is a
    continuation byte; m[len(content)] = number of characters. int64.
    Raises ValueError if `content` is not valid UTF-8."""
    content = bytes(content)
    if not utf8_roundtrip_ok(content):
        raise ValueError("content is not valid UTF-8")
    mask = char_start_mask(content)
    n_chars = int(mask.sum())
    out = np.full(len(content) + 1, -1, dtype=np.int64)
    out[:-1][mask] = np.arange(n_chars, dtype=np.int64)
    out[-1] = n_chars
    return out


def chars_to_bytes(text: str, char_offsets) -> np.ndarray:
    """Convert character offsets (0..len(text)) into byte offsets."""
    m = char_to_byte_map(text)
    c = np.asarray(char_offsets, dtype=np.int64)
    if c.size and (c.min() < 0 or c.max() > len(text)):
        raise IndexError("character offset out of range")
    return m[c]


def bytes_to_chars(content: bytes, byte_offsets) -> np.ndarray:
    """Convert byte offsets (0..len(content)) into character offsets. Raises
    ValueError for an offset that falls inside a multi-byte character."""
    m = byte_to_char_map(content)
    b = np.asarray(byte_offsets, dtype=np.int64)
    if b.size and (b.min() < 0 or b.max() > len(content)):
        raise IndexError("byte offset out of range")
    out = m[b]
    if out.size and (out < 0).any():
        raise ValueError("byte offset falls inside a multi-byte UTF-8 character")
    return out


def is_char_boundary(content: bytes, byte_offset: int) -> bool:
    """True iff byte_offset is a valid character boundary of `content`
    (0..len(content) inclusive, and not on a continuation byte)."""
    content = bytes(content)
    if byte_offset == len(content):
        return True
    if not 0 <= byte_offset < len(content):
        return False
    return (content[byte_offset] & 0xC0) != 0x80


def span_roundtrip_problems(content: bytes, spans) -> list:
    """spans: iterable of (start_byte, end_byte, expected_text). Returns one
    problem string per span whose byte slice is not a character-aligned slice
    decoding to exactly expected_text (the check build_prose_structure.py runs
    per span, reusable by any stream)."""
    content = bytes(content)
    problems = []
    for start, end, expected in spans:
        start, end = int(start), int(end)
        if not (0 <= start <= end <= len(content)):
            problems.append(f"span [{start}, {end}) outside [0, {len(content)}]")
            continue
        if not (is_char_boundary(content, start) and is_char_boundary(content, end)):
            problems.append(f"span [{start}, {end}) cuts a multi-byte character")
            continue
        try:
            got = content[start:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"span [{start}, {end}) is not valid UTF-8 ({exc})")
            continue
        if got != expected:
            problems.append(f"span [{start}, {end}) decodes to {got!r}, expected {expected!r}")
    return problems
