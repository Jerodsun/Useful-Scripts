#!/usr/bin/env python3
"""
Subtitle (SRT) Timestamp Adjuster

This script adjusts the timestamps in SRT files by a specified amount of time (earlier or later).

Functions:
    adjust_srt_times(filename, seconds, output_file=None):
        Adjusts the timestamps in the given SRT file by the specified number of seconds.
    adjust_time(time_str, delta):
        Helper function to adjust a single timestamp by the given timedelta.

Usage:
    python subtitle_adjuster.py <filename> <adjustment> [output_file]

Arguments:
    filename: The path to the SRT file to be adjusted.
    adjustment: Time adjustment in seconds (positive for later, negative for earlier).
                Can be a floating-point number.
    output_file: (Optional) Custom output filename. If not provided, defaults to 
                 "<original_filename>_adjusted.srt".

Examples:
    # Make subtitles appear 2.5 seconds earlier
    python subtitle_adjuster.py movie.srt -2.5
    
    # Make subtitles appear 1.75 seconds later with custom output file
    python subtitle_adjuster.py movie.srt 1.75 movie_fixed.srt
"""

import sys
import re
import os
from datetime import timedelta, datetime
from pathlib import Path


def adjust_time(time_str, delta):
    """
    Adjust a timestamp by the specified timedelta.

    Args:
        time_str (str): Timestamp string in the format "HH:MM:SS,mmm"
        delta (timedelta): Time adjustment to apply

    Returns:
        str: Adjusted timestamp in the same format
    """
    time_format = "%H:%M:%S,%f"
    try:
        time_obj = datetime.strptime(time_str, time_format)
        adjusted_time = (
            time_obj - delta if delta.total_seconds() > 0 else time_obj + abs(delta)
        )

        # Handle negative timestamps (prevent them)
        if adjusted_time.day < datetime(1900, 1, 1).day:
            return "00:00:00,000"

        return adjusted_time.strftime(time_format)[:-3]
    except ValueError as e:
        print(f"Warning: Could not parse timestamp '{time_str}'. Skipping. Error: {e}")
        return time_str


def adjust_srt_times(filename, seconds, output_file=None):
    """
    Adjust all timestamps in an SRT file by the specified number of seconds.

    Args:
        filename (str): Path to the input SRT file
        seconds (float): Number of seconds to adjust (positive for later, negative for earlier)
        output_file (str, optional): Custom output filename

    Returns:
        str: Path to the created output file
    """
    # Determine if we're making subtitles appear earlier or later
    earlier = seconds > 0
    adjustment_type = "later" if seconds > 0 else "earlier"
    delta = timedelta(seconds=abs(seconds))

    # Set up the output filename
    if not output_file:
        input_path = Path(filename)
        output_file = str(input_path.with_stem(f"{input_path.stem}_adjusted"))

    # Regular expression for timestamp lines
    time_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    )
    subtitle_count = 0
    adjusted_count = 0

    try:
        # Check if input file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Input file not found: {filename}")

        # Read the input file
        with open(filename, "r", encoding="utf-8-sig") as file:
            lines = file.readlines()

        # Create the output file
        with open(output_file, "w", encoding="utf-8") as file:
            for line in lines:
                # Check if the line contains timestamps
                match = time_pattern.match(line.strip())
                if match:
                    subtitle_count += 1
                    start_time, end_time = match.groups()

                    # Adjust timestamps
                    new_start_time = (
                        adjust_time(start_time, delta)
                        if earlier
                        else adjust_time(start_time, -delta)
                    )
                    new_end_time = (
                        adjust_time(end_time, delta)
                        if earlier
                        else adjust_time(end_time, -delta)
                    )

                    file.write(f"{new_start_time} --> {new_end_time}\n")
                    adjusted_count += 1
                else:
                    file.write(line)

        # Report success
        print(
            f"Successfully adjusted {adjusted_count} timestamps ({adjustment_type} by {abs(seconds)} seconds)"
        )
        print(f"Output saved to: {output_file}")
        return output_file

    except UnicodeDecodeError:
        # Try with different encodings
        for encoding in ["latin-1", "iso-8859-1", "windows-1252"]:
            try:
                with open(filename, "r", encoding=encoding) as file:
                    lines = file.readlines()

                with open(output_file, "w", encoding="utf-8") as file:
                    for line in lines:
                        match = time_pattern.match(line.strip())
                        if match:
                            subtitle_count += 1
                            start_time, end_time = match.groups()
                            new_start_time = (
                                adjust_time(start_time, delta)
                                if earlier
                                else adjust_time(start_time, -delta)
                            )
                            new_end_time = (
                                adjust_time(end_time, delta)
                                if earlier
                                else adjust_time(end_time, -delta)
                            )
                            file.write(f"{new_start_time} --> {new_end_time}\n")
                            adjusted_count += 1
                        else:
                            file.write(line)

                print(
                    f"Successfully adjusted {adjusted_count} timestamps using {encoding} encoding"
                )
                print(f"Output saved to: {output_file}")
                return output_file
            except Exception as e:
                continue

        # If we get here, none of the encodings worked
        print(
            f"Error: Could not decode the input file. Please check the file encoding."
        )
        return None

    except Exception as e:
        print(f"Error adjusting subtitles: {str(e)}")
        return None


def validate_file(filename):
    """
    Validate that the file exists and appears to be an SRT file.

    Args:
        filename (str): Path to the file to validate

    Returns:
        bool: True if the file is valid, False otherwise
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return False

    # Check extension
    if not filename.lower().endswith(".srt"):
        print(f"Warning: File '{filename}' does not have an .srt extension.")

    # Basic content check - look for timestamp pattern in first 20 lines
    time_pattern = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")
    try:
        with open(filename, "r", encoding="utf-8-sig") as file:
            for _ in range(20):
                line = file.readline()
                if not line:
                    break
                if time_pattern.search(line):
                    return True

        print(f"Warning: File '{filename}' does not appear to be a valid SRT file.")
        user_input = input("Continue anyway? (y/n): ")
        return user_input.lower() == "y"
    except Exception:
        # Try with different encodings
        try:
            with open(filename, "r", encoding="latin-1") as file:
                for _ in range(20):
                    line = file.readline()
                    if not line:
                        break
                    if time_pattern.search(line):
                        return True
            return False
        except Exception as e:
            print(f"Error validating file: {str(e)}")
            return False


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <filename> <adjustment> [output_file]")
        print(
            "  adjustment: Time in seconds (negative for earlier, positive for later)"
        )
        sys.exit(1)

    filename = sys.argv[1]

    try:
        seconds = float(sys.argv[2])
    except ValueError:
        print(f"Error: Invalid time adjustment '{sys.argv[2]}'. Must be a number.")
        sys.exit(1)

    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate the input file
    if not validate_file(filename):
        sys.exit(1)

    # Adjust the subtitles
    result = adjust_srt_times(filename, seconds, output_file)
    if result:
        print("Subtitle adjustment completed successfully.")
    else:
        print("Subtitle adjustment failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
