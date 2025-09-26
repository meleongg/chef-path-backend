#!/usr/bin/env python3
"""
Database Management Script for ChefPath Backend

This is a convenient wrapper script that provides easy access to all database operations.

Usage:
    python manage_db.py clear              # Clear database
    python manage_db.py seed               # Seed with sample data
    python manage_db.py seed --clear       # Clear then seed
    python manage_db.py view               # View all data
    python manage_db.py view --summary     # View summary only
    python manage_db.py reset              # Clear and seed (full reset)
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List


def run_script(script_name: str, args: Optional[List[str]] = None):
    """Run a script in the scripts directory"""
    script_path = Path(__file__).parent / "scripts" / f"{script_name}.py"

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Script failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="ChefPath Database Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_db.py clear                    # Clear all data
  python manage_db.py seed                     # Add sample data
  python manage_db.py seed --clear             # Clear then add sample data
  python manage_db.py view                     # View all data
  python manage_db.py view --summary           # View summary only
  python manage_db.py view --table users      # View users table only
  python manage_db.py reset                    # Full reset (clear + seed)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear database")
    clear_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # Seed command
    seed_parser = subparsers.add_parser("seed", help="Seed database with sample data")
    seed_parser.add_argument(
        "--clear", action="store_true", help="Clear database first"
    )

    # View command
    view_parser = subparsers.add_parser("view", help="View database contents")
    view_parser.add_argument("--summary", action="store_true", help="Show summary only")
    view_parser.add_argument(
        "--table",
        choices=["users", "recipes", "weekly_plans", "user_recipe_progress"],
        help="View specific table",
    )
    view_parser.add_argument("--limit", type=int, help="Limit records displayed")

    # Reset command (convenience)
    reset_parser = subparsers.add_parser(
        "reset", help="Full reset: clear and seed database"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    success = True

    if args.command == "clear":
        script_args = []
        if hasattr(args, "force") and args.force:
            script_args.append("--force")
        success = run_script("clear_database", script_args)

    elif args.command == "seed":
        script_args = []
        if hasattr(args, "clear") and args.clear:
            script_args.append("--clear")
        success = run_script("seed_database", script_args)

    elif args.command == "view":
        script_args = []
        if hasattr(args, "summary") and args.summary:
            script_args.append("--summary")
        if hasattr(args, "table") and args.table:
            script_args.extend(["--table", args.table])
        if hasattr(args, "limit") and args.limit:
            script_args.extend(["--limit", str(args.limit)])
        success = run_script("view_database", script_args)

    elif args.command == "reset":
        print("🔄 Performing full database reset...")
        success = run_script("seed_database", ["--clear"])

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
