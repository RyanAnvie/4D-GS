# Regression Tests

Open `run_checks.py` in the repository root and press F5 in Spyder with the
`envtorch` kernel. Logs and the combined summary are written under `outputs/`.

Each `test_*.py` here can also be run directly with F5. Every test resolves the
repository root from its own file location, so imports and bundled assets do
not depend on Spyder's working directory. The suite includes CPU and CUDA
checks. Its default viewer tests inspect data and state without opening windows.

The unified runner starts each test in a fresh Python process using the current
interpreter. Tests run from the new output directory to check working-directory
independence. Training code and the saved example models are not modified.
