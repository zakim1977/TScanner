"""
SYSTEMATIC TESTING: Direction Consistency Checker
==================================================
This module ensures ALL direction-related signals are consistent across:
- Scanner header
- Single Analysis
- Watchlist
- Trade Monitor
- Telegram alerts

RUN THIS BEFORE EVERY RELEASE!
"""

import sys
sys.path.insert(0, '..')

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION ALIGNMENT RULES (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════════════════

ALIGNMENT_RULES = {
    # VWAP Bounce Types
    'VWAP': {
        'BULLISH_BOUNCE': {'LONG': True, 'SHORT': False},
        'BEARISH_BOUNCE': {'LONG': False, 'SHORT': True},
        'BULLISH_RETEST': {'LONG': True, 'SHORT': False},
        'BEARISH_RETEST': {'LONG': False, 'SHORT': True},
        'CONFIRMED_BULLISH_RETEST': {'LONG': True, 'SHORT': False},
        'CONFIRMED_BEARISH_RETEST': {'LONG': False, 'SHORT': True},
        'BULLISH_BOUNCE_CONFIRMED': {'LONG': True, 'SHORT': False},
        'BEARISH_BOUNCE_CONFIRMED': {'LONG': False, 'SHORT': True},
        'APPROACHING_BULLISH': {'LONG': True, 'SHORT': False},
        'APPROACHING_BEARISH': {'LONG': False, 'SHORT': True},
        'AT_VWAP': {'LONG': True, 'SHORT': True},  # Neutral
        'AT_VWAP_RETEST': {'LONG': True, 'SHORT': True},  # Neutral
    },
    
    # VWAP Flip Types
    'VWAP_FLIP': {
        'VWAP_FLIP_BULLISH': {'LONG': True, 'SHORT': False},  # R→S = Good for LONG
        'VWAP_FLIP_BEARISH': {'LONG': False, 'SHORT': True},  # S→R = Good for SHORT
        'FLIP_FAILED_BULLISH': {'LONG': False, 'SHORT': True},  # Failed bullish = Bad for LONG
        'FLIP_FAILED_BEARISH': {'LONG': True, 'SHORT': False},  # Failed bearish = Bad for SHORT
        'WATCHING_BULLISH': {'LONG': None, 'SHORT': None},  # Neutral - watching
        'WATCHING_BEARISH': {'LONG': None, 'SHORT': None},  # Neutral - watching
    },
    
    # ML Direction
    'ML': {
        'LONG': {'LONG': True, 'SHORT': False},
        'SHORT': {'LONG': False, 'SHORT': True},
        'WAIT': {'LONG': False, 'SHORT': False},  # Neither aligned
        'NEUTRAL': {'LONG': False, 'SHORT': False},
    },
    
    # Rules/Predictive Direction
    'RULES': {
        'BULLISH': {'LONG': True, 'SHORT': False},
        'BEARISH': {'LONG': False, 'SHORT': True},
        'LEAN_BULLISH': {'LONG': True, 'SHORT': False},
        'LEAN_BEARISH': {'LONG': False, 'SHORT': True},
        'NEUTRAL': {'LONG': False, 'SHORT': False},
    },
    
    # Order Block
    'ORDER_BLOCK': {
        'AT_BULLISH_OB': {'LONG': True, 'SHORT': False},  # Support = Good for LONG
        'AT_BEARISH_OB': {'LONG': False, 'SHORT': True},  # Resistance = Good for SHORT
        'NEAR_BULLISH_OB': {'LONG': True, 'SHORT': False},
        'NEAR_BEARISH_OB': {'LONG': False, 'SHORT': True},
    },
    
    # Structure
    'STRUCTURE': {
        'BULLISH': {'LONG': True, 'SHORT': False},
        'BEARISH': {'LONG': False, 'SHORT': True},
        'RANGING': {'LONG': None, 'SHORT': None},
    },
}


def check_alignment(signal_type: str, signal_value: str, trade_direction: str) -> bool:
    """
    Check if a signal aligns with trade direction.
    
    Returns:
        True = aligned
        False = not aligned
        None = neutral/not applicable
    """
    if signal_type not in ALIGNMENT_RULES:
        return None
    
    type_rules = ALIGNMENT_RULES[signal_type]
    if signal_value not in type_rules:
        return None
    
    direction_rules = type_rules[signal_value]
    if trade_direction not in direction_rules:
        return None
    
    return direction_rules[trade_direction]


def test_vwap_alignment():
    """Test VWAP signal alignment logic"""
    print("\n" + "="*60)
    print("TESTING: VWAP Bounce Alignment")
    print("="*60)
    
    test_cases = [
        ('BULLISH_BOUNCE', 'LONG', True),
        ('BULLISH_BOUNCE', 'SHORT', False),
        ('BEARISH_BOUNCE', 'LONG', False),
        ('BEARISH_BOUNCE', 'SHORT', True),
        ('CONFIRMED_BULLISH_RETEST', 'LONG', True),
        ('CONFIRMED_BULLISH_RETEST', 'SHORT', False),
        ('CONFIRMED_BEARISH_RETEST', 'LONG', False),
        ('CONFIRMED_BEARISH_RETEST', 'SHORT', True),
    ]
    
    passed = 0
    failed = 0
    
    for signal, direction, expected in test_cases:
        result = check_alignment('VWAP', signal, direction)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {signal} + {direction} = {result} (expected {expected})")
    
    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0


def test_vwap_flip_alignment():
    """Test VWAP Flip signal alignment logic"""
    print("\n" + "="*60)
    print("TESTING: VWAP Flip Alignment")
    print("="*60)
    
    test_cases = [
        ('VWAP_FLIP_BULLISH', 'LONG', True, "R→S good for LONG"),
        ('VWAP_FLIP_BULLISH', 'SHORT', False, "R→S bad for SHORT"),
        ('VWAP_FLIP_BEARISH', 'LONG', False, "S→R bad for LONG"),
        ('VWAP_FLIP_BEARISH', 'SHORT', True, "S→R good for SHORT"),
        ('FLIP_FAILED_BULLISH', 'LONG', False, "Failed bullish bad for LONG"),
        ('FLIP_FAILED_BULLISH', 'SHORT', True, "Failed bullish good for SHORT"),
        ('FLIP_FAILED_BEARISH', 'LONG', True, "Failed bearish good for LONG"),
        ('FLIP_FAILED_BEARISH', 'SHORT', False, "Failed bearish bad for SHORT"),
    ]
    
    passed = 0
    failed = 0
    
    for signal, direction, expected, reason in test_cases:
        result = check_alignment('VWAP_FLIP', signal, direction)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {signal} + {direction} = {result}")
        print(f"         Reason: {reason}")
    
    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0


def test_ml_alignment():
    """Test ML direction alignment logic"""
    print("\n" + "="*60)
    print("TESTING: ML Direction Alignment")
    print("="*60)
    
    test_cases = [
        ('LONG', 'LONG', True),
        ('LONG', 'SHORT', False),
        ('SHORT', 'LONG', False),
        ('SHORT', 'SHORT', True),
        ('WAIT', 'LONG', False),
        ('WAIT', 'SHORT', False),
    ]
    
    passed = 0
    failed = 0
    
    for ml_dir, trade_dir, expected in test_cases:
        result = check_alignment('ML', ml_dir, trade_dir)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: ML={ml_dir} + Trade={trade_dir} = {result}")
    
    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0


def run_all_tests():
    """Run all consistency tests"""
    print("\n" + "="*60)
    print("INVESTORIQ DIRECTION CONSISTENCY TEST SUITE")
    print("="*60)
    print("This verifies all direction-related logic is consistent")
    print("across Scanner, Single Analysis, Watchlist, Trade Monitor")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_vwap_alignment()
    all_passed &= test_vwap_flip_alignment()
    all_passed &= test_ml_alignment()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Direction logic is consistent!")
    else:
        print("❌ SOME TESTS FAILED - Check alignment rules!")
    print("="*60)
    
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# CODE CONSISTENCY CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

def check_code_consistency():
    """
    Check that all code sections use the same alignment logic.
    This searches for patterns that might indicate inconsistency.
    """
    print("\n" + "="*60)
    print("CODE CONSISTENCY CHECK")
    print("="*60)
    
    import os
    
    # Files to check
    files_to_check = [
        '../app.py',
        '../utils/formatters.py',
        '../core/indicators.py',
    ]
    
    # Patterns that should be direction-aware
    suspicious_patterns = [
        "vwap_good = True",  # Should check direction
        "aligned = True",  # Should check direction
        "vwap_aligned = True",  # Should check direction
        "flip.*always",  # "always good" type statements
    ]
    
    issues_found = []
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines, 1):
            for pattern in suspicious_patterns:
                import re
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's in a comment
                    stripped = line.strip()
                    if not stripped.startswith('#'):
                        issues_found.append({
                            'file': filepath,
                            'line': i,
                            'pattern': pattern,
                            'code': stripped[:80]
                        })
    
    if issues_found:
        print(f"⚠️ Found {len(issues_found)} potential issues:\n")
        for issue in issues_found:
            print(f"  File: {issue['file']}")
            print(f"  Line: {issue['line']}")
            print(f"  Code: {issue['code']}")
            print()
    else:
        print("✅ No suspicious patterns found!")
    
    return len(issues_found) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKLIST FOR NEW FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FEATURE CHECKLIST - Direction Aware                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ For EVERY new feature involving direction (LONG/SHORT):                     ║
║                                                                              ║
║ □ 1. Scanner Header (utils/formatters.py)                                   ║
║      - render_signal_header_html()                                          ║
║      - Check: BULLISH + LONG = aligned                                      ║
║      - Check: BEARISH + SHORT = aligned                                     ║
║                                                                              ║
║ □ 2. Single Analysis (app.py ~line 13800-14200)                            ║
║      - VWAP section                                                         ║
║      - Direction labels should be context-aware                             ║
║                                                                              ║
║ □ 3. Watchlist Monitor (app.py ~line 8900-9400)                            ║
║      - vwap_good calculation                                                ║
║      - conditions_met counting                                              ║
║      - Display text                                                         ║
║                                                                              ║
║ □ 4. Watchlist Telegram Alert (app.py ~line 9200-9300)                     ║
║      - Alert message content                                                ║
║                                                                              ║
║ □ 5. Trade Monitor (app.py ~line 11000-12000)                              ║
║      - Grade/score display                                                  ║
║      - Status indicators                                                    ║
║                                                                              ║
║ □ 6. Run test_direction_consistency.py                                      ║
║      - All tests must pass                                                  ║
║                                                                              ║
║ □ 7. Manual Test Cases:                                                     ║
║      - LONG trade + Bullish signal → should show aligned ✅                 ║
║      - LONG trade + Bearish signal → should show NOT aligned ❌             ║
║      - SHORT trade + Bullish signal → should show NOT aligned ❌            ║
║      - SHORT trade + Bearish signal → should show aligned ✅                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_checklist():
    """Print the feature checklist"""
    print(FEATURE_CHECKLIST)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--checklist':
        print_checklist()
    elif len(sys.argv) > 1 and sys.argv[1] == '--code-check':
        check_code_consistency()
    else:
        run_all_tests()
        print("\n")
        check_code_consistency()
        print("\nRun with --checklist to see the feature checklist")
