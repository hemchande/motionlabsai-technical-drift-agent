# Temporary Directory Fix

## Issue

When running `python main.py`, you may encounter:
```
FileNotFoundError: [Errno 2] No usable temporary directory found
```

This happens because `vision_agents` tries to use `tempfile.gettempdir()` at import time, and sometimes the default temp directory isn't accessible.

## Solution

The fix has been added to `main.py` to:
1. Check if the default temp directory is usable
2. Fall back to `/tmp` if the default isn't available
3. Create a local `.tmp` directory as a last resort

## How It Works

Before importing `vision_agents`, the code:
1. Tries to get the default temp directory
2. Checks if it exists and is writable
3. If not, sets `TMPDIR` environment variable to `/tmp`
4. If `/tmp` also fails, creates a local `.tmp` directory

## Testing

To verify the fix works:
```bash
cd cvMLAgent
.venv/bin/python main.py --help
```

If you still see the error, you can manually set the temp directory:
```bash
export TMPDIR=/tmp
.venv/bin/python main.py
```

## Alternative Solutions

If the issue persists:

1. **Set TMPDIR in your shell:**
   ```bash
   export TMPDIR=/tmp
   ```

2. **Add to your `.env` file:**
   ```
   TMPDIR=/tmp
   ```

3. **Create temp directory manually:**
   ```bash
   mkdir -p /tmp
   chmod 1777 /tmp
   ```
