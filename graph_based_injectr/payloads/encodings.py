"""
Payload Encoding Techniques for AutoInjector

This module provides functions to encode and obfuscate payloads to bypass
input filters and detection mechanisms. Encoding can help evade:
    - Keyword blocklists
    - Pattern matching filters
    - Input sanitization

Available encodings:
    - Base64
    - ROT13
    - Hex
    - Unicode escapes
    - Character substitution
    - Whitespace injection
"""

import base64
import codecs
from typing import List


def encode_base64(text: str) -> str:
    """
    Encode text as Base64.
    
    Base64 encoding can bypass simple keyword filters that look
    for specific strings in the input.
    
    Args:
        text: The text to encode
        
    Returns:
        Base64-encoded string
    """
    return base64.b64encode(text.encode()).decode()


def decode_base64(encoded: str) -> str:
    """
    Decode Base64 text.
    
    Args:
        encoded: Base64-encoded string
        
    Returns:
        Decoded text
    """
    return base64.b64decode(encoded.encode()).decode()


def encode_rot13(text: str) -> str:
    """
    Encode text using ROT13 cipher.
    
    ROT13 is a simple letter substitution cipher that replaces
    each letter with the one 13 positions after it.
    
    Args:
        text: The text to encode
        
    Returns:
        ROT13-encoded string
    """
    return codecs.encode(text, 'rot_13')


def decode_rot13(encoded: str) -> str:
    """
    Decode ROT13 text.
    
    Args:
        encoded: ROT13-encoded string
        
    Returns:
        Decoded text
    """
    return codecs.decode(encoded, 'rot_13')


def encode_hex(text: str) -> str:
    """
    Encode text as hexadecimal.
    
    Args:
        text: The text to encode
        
    Returns:
        Hex-encoded string
    """
    return text.encode().hex()


def decode_hex(encoded: str) -> str:
    """
    Decode hexadecimal text.
    
    Args:
        encoded: Hex-encoded string
        
    Returns:
        Decoded text
    """
    return bytes.fromhex(encoded).decode()


def encode_unicode_escapes(text: str) -> str:
    """
    Encode text using Unicode escape sequences.
    
    Converts each character to its Unicode escape form (\\uXXXX).
    
    Args:
        text: The text to encode
        
    Returns:
        Unicode-escaped string
    """
    return "".join(f"\\u{ord(c):04x}" for c in text)


def encode_reverse(text: str) -> str:
    """
    Reverse the text.
    
    Simple obfuscation that reverses the character order.
    
    Args:
        text: The text to reverse
        
    Returns:
        Reversed string
    """
    return text[::-1]


def inject_whitespace(text: str, char: str = "\u200b") -> str:
    """
    Inject invisible characters between letters.
    
    Uses zero-width space or other invisible characters to break
    up keyword patterns while keeping text visually the same.
    
    Args:
        text: The text to process
        char: The character to inject (default: zero-width space)
        
    Returns:
        Text with injected characters
    """
    return char.join(text)


def encode_leetspeak(text: str) -> str:
    """
    Convert text to leetspeak.
    
    Substitutes letters with similar-looking numbers/symbols
    to bypass keyword filters.
    
    Args:
        text: The text to convert
        
    Returns:
        Leetspeak version
    """
    leet_map = {
        'a': '4', 'e': '3', 'i': '1', 'o': '0',
        's': '5', 't': '7', 'l': '1', 'A': '4',
        'E': '3', 'I': '1', 'O': '0', 'S': '5',
        'T': '7', 'L': '1',
    }
    return "".join(leet_map.get(c, c) for c in text)


def encode_mixed_case(text: str) -> str:
    """
    Randomize case of letters.
    
    Alternates upper/lowercase to potentially bypass case-sensitive filters.
    
    Args:
        text: The text to process
        
    Returns:
        Mixed-case version
    """
    result = []
    for i, c in enumerate(text):
        if i % 2 == 0:
            result.append(c.upper())
        else:
            result.append(c.lower())
    return "".join(result)


def add_separator(text: str, separator: str = ".") -> str:
    """
    Add separators between characters.
    
    Can break up keyword patterns while remaining readable.
    
    Args:
        text: The text to process
        separator: Character to insert between letters
        
    Returns:
        Text with separators
    """
    return separator.join(text)


def encode_with_comments(text: str) -> str:
    """
    Inject comment-like syntax into the text.
    
    Adds characters that might be stripped by processors.
    
    Args:
        text: The text to process
        
    Returns:
        Text with comment markers
    """
    # Add HTML-style comments between words
    words = text.split()
    return "<!-- -->".join(words)


def get_all_encodings(text: str) -> List[dict]:
    """
    Get all encoded versions of the text.
    
    Returns a list of dictionaries with encoding name and result.
    
    Args:
        text: The text to encode
        
    Returns:
        List of {"encoding": name, "result": encoded_text}
    """
    encodings = [
        ("base64", encode_base64(text)),
        ("rot13", encode_rot13(text)),
        ("hex", encode_hex(text)),
        ("unicode_escapes", encode_unicode_escapes(text)),
        ("reverse", encode_reverse(text)),
        ("zero_width_space", inject_whitespace(text)),
        ("leetspeak", encode_leetspeak(text)),
        ("mixed_case", encode_mixed_case(text)),
        ("dotted", add_separator(text)),
        ("with_comments", encode_with_comments(text)),
    ]
    
    return [{"encoding": name, "result": result} for name, result in encodings]
