#!/usr/bin/env python3
"""
Script to run or continue a Python process within a specified time window.
Usage:
  python run_at_night.py "python /path/to/your_script.py"
Start and end times can be configured in the config() function.

Testing and debugging this script:
  python run_at_night.py test
This will run a test script (test.py) that prints iterations every second. The
time window will be set to start immediately and end after 10 seconds.
"""

import os
import sys
import time
import subprocess
import psutil

PID_FILE = "/tmp/run_at_night.pid"

def config():
  from datetime import datetime
  default = {
    "start": "18:00:00",
    "end": "08:00:00",
    "sleep": 60,
    "command": sys.argv[1]
  }

  if sys.argv[1] == "test":
    now = datetime.now()
    default["start"] = now.strftime("%H:%M:%S")
    end_time = now.timestamp() + 10
    default["end"] = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
    default["command"] = "python test.py"
    default["sleep"] = 1

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
  print(f"{now_str()} - Starting script using '{cfg['command']}' ...")

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
    proc = psutil.Process(pid)
    msg = f"{now_str()} - Resuming script (PID {pid})..."
    print(msg)
    proc.resume()
    print(f"Script resumed (PID {pid})")
  except psutil.NoSuchProcess:
    print(f"Process {pid} no longer exists.")
  except Exception as e:
    print(f"Error resuming script: {e}", file=sys.stderr)


def pause_script(pid):
    """Pause (SIGSTOP) a running process using psutil for macOS/Linux compatibility."""
    try:
        proc = psutil.Process(pid)
        # Check if process is already paused
        if proc.status() == psutil.STATUS_STOPPED:
            print(f"{now_str()} - Script is paused (PID {pid})")
            return

        msg = f"{now_str()} - Pausing script (PID {pid})..."
        print(msg)
        
        # Send pause signal
        proc.suspend() 
        print(f"{now_str()} - Script paused (PID {pid})")
        
    except psutil.NoSuchProcess:
        print(f"{now_str()} - Process {pid} no longer exists.")
    except Exception as e:
        print(f"{now_str()} - Error pausing script: {e}", file=sys.stderr)


def cleanup():
    """Clean up the PID file."""
    if os.path.exists(PID_FILE):
      os.remove(PID_FILE)
      print(f"Removed PID file: {PID_FILE}")
    else:
      print("No PID file to remove.")


def main(cfg):
  """Main loop to manage the script based on time window."""

  msg = f"Command\n  {cfg['command']}\nconfigured to run between "
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
              # Process exists; check if it's paused and resume if needed
              # Note: There's no reliable way to check if a process is paused,
              # so we just send SIGCONT which is harmless if already running
              try:
                  proc = psutil.Process(pid)
                  if proc.status() == psutil.STATUS_STOPPED:
                      resume_script(pid)
                  else:
                      print(f"{now_str()} - Process is already running (PID {pid}).")
              except psutil.NoSuchProcess:
                  print(f"{now_str()} - Process (PID {pid}) no longer exists. Cleaning up PID file.")
                  cleanup()
      else:
          # We should NOT be running
          if process_running:
              pause_script(pid)
          else:
              print(f"{now_str()} - Outside time window and no process to pause.")

      print(f"{now_str()} - Waiting {cfg['sleep']} seconds for next check if script should be paused or unpaused ...\n")
      time.sleep(cfg["sleep"])
      

  except KeyboardInterrupt:
    print("\nKeyboard interrupt. Shutting down...")
    pid = get_process_pid()
    if pid and is_process_running(pid):
      print(f"Stopping process (PID {pid})...")
      try:
        proc = psutil.Process(pid)
        proc.terminate()
        time.sleep(1)
        if proc.is_running():
          proc.kill()
      except Exception as e:
        pass

    cleanup()

    print("Exiting with code 0.")
    sys.exit(0)


if __name__ == "__main__":
  main(config())
