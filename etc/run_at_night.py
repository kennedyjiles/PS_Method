#!/usr/bin/env python3
"""
Script to run or continue a Python process within a specified time window.
Usage:
  python run_at_night.py "/path/to/your_script.py"
Start and end times can be configured in the config() function.
"""

import os
import sys
import time
import signal
import subprocess


PID_FILE = "/tmp/dipole_script.pid"

def config(test=False):
  from datetime import datetime
  default = {
    "start": "18:00:00",
    "end": "08:00:00",
    "sleep": 1,
    "command": sys.argv[1]
  }

  if test:
    now = datetime.now()
    default["start"] = now.strftime("%H:%M:%S")
    end_time = now.timestamp() + 20
    default["end"] = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

  return default


def now_str():
  from datetime import datetime
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now(string=False):
  from datetime import datetime
  now = datetime.now()
  return now.hour, now.minute, now.second


def is_within_time_window(cfg):
    """Check if current time is within the allowed time window."""
    current_hour, current_minute, current_second = now()
    START_HOUR, START_MINUTE, START_SECOND = map(int, cfg['start'].split(':'))
    END_HOUR, END_MINUTE, END_SECOND = map(int, cfg['end'].split(':'))

    # Convert times to seconds since midnight for easier comparison
    current_time_secs = current_hour * 3600 + current_minute * 60 + current_second
    start_time_secs = START_HOUR * 3600 + START_MINUTE * 60 + START_SECOND
    end_time_secs = END_HOUR * 3600 + END_MINUTE * 60 + END_SECOND

    # Handle time windows that cross midnight
    if start_time_secs > end_time_secs:
      # Time window crosses midnight (e.g., 22:00:00 to 06:00:00)
      return current_time_secs >= start_time_secs or current_time_secs < end_time_secs
    else:
      # Normal time window (e.g., 09:00:00 to 17:00:00)
      return start_time_secs <= current_time_secs < end_time_secs


def get_process_pid():
    """Read the PID from the PID file if it exists."""
    if os.path.exists(PID_FILE):
      try:
        with open(PID_FILE, 'r') as f:
          pid = int(f.read().strip())
          return pid
      except (ValueError, IOError):
        return None
    return None


def is_process_running(pid):
  """Check if a process with the given PID is running."""
  if pid is None:
    return False
  try:
    # Send signal 0 to check if process exists
    os.kill(pid, 0)
    return True
  except OSError:
    return False


def start_script(cfg):
  """Start the script as a background process."""
  print(f"{now_str()} - Starting script with {cfg['command']}...")

  try:
    # Start the process in the background
    process = subprocess.Popen(
        cfg["command"].split(),
        stdout=None,  # Inherit stdout to print to console
        stderr=None,  # Inherit stderr to print to console
        preexec_fn=os.setpgrp  # Create new process group
    )

    # Save PID to file
    with open(PID_FILE, 'w') as f:
      f.write(str(process.pid))

    msg = f"{now_str()} - Script started with PID {process.pid}."
    print(msg)
    return process.pid
  except Exception as e:
    print(f"{now_str()} - Error starting script: {e}", file=sys.stderr)
    return None


def resume_script(pid):
  """Resume (SIGCONT) a paused process."""
  try:
    msg = f"{now_str()} - Resuming script (PID {pid})..."
    print(msg)
    os.kill(pid, signal.SIGCONT)
    print(f"Script resumed (PID {pid})")
  except OSError as e:
    print(f"Error resuming script: {e}", file=sys.stderr)


def pause_script(pid):
    """Pause (SIGSTOP) a running process."""

    try:
      with open(f"/proc/{pid}/status", 'r') as f:
        status = f.read()
      if "State:\tT (stopped)" in status:
        print(f"Script is already paused (PID {pid})")
        return
    except (FileNotFoundError, IOError):
      print(f"Could not read /proc/{pid}/status to check if process is paused.")
      exit(1)

    try:
        # Check if process is already paused
      msg = f"{now_str()} - Pausing script (PID {pid})..."
      print(msg)
      os.kill(pid, signal.SIGSTOP)
      print(f"Script paused (PID {pid})")
    except OSError as e:
      print(f"Error pausing script: {e}", file=sys.stderr)


def cleanup():
    """Clean up the PID file."""
    if os.path.exists(PID_FILE):
      os.remove(PID_FILE)
      print(f"Removed PID file: {PID_FILE}")
    else:
      print("No PID file to remove.")


def main(cfg):
  """Main loop to manage the script based on time window."""

  msg = f"Command\n  {' '.join(cfg['command'])}\nconfigured to run between "
  msg += f"{cfg['start']} and {cfg['end']}"
  msg += " and paused outside this window."
  print(msg)

  print(f"Checking every {cfg['sleep']} seconds")
  print("-" * 60)

  # Clean up old PID file if it exists
  if os.path.exists(PID_FILE):
      old_pid = get_process_pid()
      if old_pid and not is_process_running(old_pid):
        print(f"Removing stale PID file (PID {old_pid} not running)")
        cleanup()
      else:
        msg = f"PID file exists with running process (PID {old_pid}), will manage this process."
        print(msg)

  try:
    while True:
      in_window = is_within_time_window(cfg)
      pid = get_process_pid()
      process_running = is_process_running(pid)

      msg = f"{now_str()} - In window: {in_window}; "
      msg += f"Process running: {process_running}"
      print(msg)

      if in_window:
        # We should be running
        if not process_running:
          # Start the script
          start_script(cfg)
        else:
          # Process exists, make sure it's not paused (try to resume)
          # Note: There's no reliable way to check if a process is paused,
          # so we just send SIGCONT which is harmless if already running
          print(f"{now_str()} - Process is already running (PID {pid}).")
      else:
        # We should NOT be running
        if process_running:
          # Pause the script
          pause_script(pid)
        else:
          print(f"{now_str()} - Outside time window and no process to pause.")

      print(f"{now_str()} - Sleeping for {cfg['sleep']} seconds...\n")
      time.sleep(cfg["sleep"])

  except KeyboardInterrupt:
    print("\nKeyboard interrupt. Shutting down...")
    pid = get_process_pid()
    if pid and is_process_running(pid):
      print(f"Stopping process (PID {pid})...")
      try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        if is_process_running(pid):
          os.kill(pid, signal.SIGKILL)
      except OSError:
        pass

    cleanup()

    print("Exiting with code 0.")
    sys.exit(0)


if __name__ == "__main__":
  #main(config())
  # Start process now and run for 20 seconds.
  main(config(test=True))
