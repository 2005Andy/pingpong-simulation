#!/usr/bin/env python3
"""Simple test script to verify the modular ball sports simulation works correctly."""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    try:
        import constants
        import ball_types
        import physics
        import racket_control
        import scenarios
        import simulation
        import visualization
        import pingpong_main
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_simulation():
    """Test a basic simulation run."""
    try:
        from scenarios import create_serve_scenario, create_table, create_net
        from simulation import simulate

        # Create a simple scenario
        initial_ball, strokes_a, strokes_b = create_serve_scenario()
        table = create_table()
        net = create_net()

        # Run a very short simulation
        result = simulate(
            initial_ball=initial_ball,
            strokes_a=strokes_a,
            strokes_b=strokes_b,
            table=table,
            net=net,
            dt=0.001,
            max_time=0.5,
            record_interval=50
        )

        print("✓ Basic simulation completed successfully")
        print(f"  - Recorded {len(result.ball_history['t'])} time steps")
        print(f"  - {result.rally_count} rallies")
        return True

    except Exception as e:
        print(f"✗ Simulation error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing modular ball sports simulation...")
    print("=" * 50)

    tests = [
        ("Module imports", test_imports),
        ("Basic simulation", test_basic_simulation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"Failed: {test_name}")

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Modular architecture is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
