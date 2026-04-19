#!/usr/bin/env python3
"""
Simple run script for our TPU Simulator project.
We made this to make it easy for anyone to test our simulator!
"""

import sys
import os

def main():
    """We offer a simple menu to run different parts of the simulator"""
    
    print("\n" + "="*60)
    print("TPU SIMULATOR - RUN MENU")
    print("="*60)
    print("\nWe made this menu to easily test different parts!\n")
    print("Options:")
    print("  1. Run main simulator (tpu_simulator.py)")
    print("  2. Run benchmarks (benchmark.py)")
    print("  3. Run comprehensive tests (test_simulator.py)")
    print("  4. Exit\n")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n[INFO] Running main TPU simulator...\n")
        os.system("python3.13 tpu_simulator.py")
    elif choice == "2":
        print("\n[INFO] Running benchmarks...\n")
        os.system("python3.13 benchmark.py")
    elif choice == "3":
        print("\n[INFO] Running comprehensive tests...\n")
        os.system("python3.13 test_simulator.py")
    elif choice == "4":
        print("\nGoodbye! Thanks for testing our TPU simulator!")
    else:
        print("\n✗ Invalid choice! Please enter 1-4")
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] We exited the menu")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("We're sorry something went wrong!")
