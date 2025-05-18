# Project Context

This project, `parakeet-mlx_guiapi`, is designed to provide a GUI and API for the `parakeet-mlx` speech-to-text library.

**Important Setup Detail:**

For this project to run correctly, the `parakeet-mlx` repository must be cloned in the **same parent directory** as the `parakeet-mlx_guiapi` repository.

For example, if your directory structure looks like this:

```
/
├── projects/
│   ├── parakeet-mlx/
│   └── parakeet-mlx_guiapi/
```

The `run.py` script in `parakeet-mlx_guiapi` automatically determines the path to `parakeet-mlx` by assuming this relative location. It adds the `parakeet-mlx` directory to the Python system path (`sys.path`) at runtime to allow importing the necessary modules.

Ensure your local setup matches this structure to avoid "No module named 'parakeet_mlx'" errors.
