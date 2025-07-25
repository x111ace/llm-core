import os, sys, shutil, argparse, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def get_env_dir(env_name: str = 'llm_core'):
    """Gets the predictable path for a *local venv* environment directory."""
    env_dir_name = f"{env_name}_venv"
    return os.path.join(PROJECT_ROOT, env_dir_name)

def _find_conda_env_path(env_name: str):
    """Parses `conda info --envs` to find the full path of a named environment."""
    try:
        is_windows = sys.platform == "win32"
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True, text=True, check=True, shell=is_windows, encoding='utf-8'
        )
        for line in result.stdout.splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            # The name is the first part, the path is the last.
            if len(parts) >= 2 and parts[0] == env_name:
                return " ".join(parts[1:]).strip() # Handle paths with spaces
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def create_env(use_conda: bool = True, env_name_str: str = 'llm_core'):
    """
    Creates a Python environment and returns (path_to_python_exe, env_type, env_path).
    """
    env_name = f"{env_name_str}_venv"

    def _check_conda():
        try:
            is_windows = sys.platform == "win32"
            subprocess.run(["conda", "info"], check=True, capture_output=True, shell=is_windows)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _create_conda():
        env_path = _find_conda_env_path(env_name)
        if env_path:
            print(f"Conda environment '{env_name}' already exists at {env_path}")
            python_exe = os.path.join(env_path, "python.exe") if sys.platform == "win32" else os.path.join(env_path, "bin", "python")
            return python_exe, "conda", env_path

        try:
            print(f"Creating Conda environment '{env_name}'...")
            is_windows = sys.platform == "win32"
            subprocess.run(
                ["conda", "create", "--name", env_name, "python=3.12", "-y"],
                check=True, shell=is_windows
            )
            print("Conda environment created successfully.")
            
            new_env_path = _find_conda_env_path(env_name)
            if new_env_path:
                python_exe = os.path.join(new_env_path, "python.exe") if sys.platform == "win32" else os.path.join(new_env_path, "bin", "python")
                return python_exe, "conda", new_env_path
            else:
                print(f"Failed to find path for new Conda env '{env_name}'.", file=sys.stderr)
                return None, None, None
        except subprocess.CalledProcessError as e:
            print(f"Error creating Conda environment: {e}.", file=sys.stderr)
            return None, None, None

    def _create_venv():
        env_path = get_env_dir(env_name_str)
        python_exe = os.path.join(env_path, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(env_path, "bin", "python")
        if os.path.exists(env_path):
            print(f"Virtual environment '{env_name}' already exists.")
            return python_exe, "venv", env_path
        try:
            print(f"Creating venv '{env_name}'...")
            subprocess.run([sys.executable, "-m", "venv", env_path], check=True)
            print("venv created successfully.")
            return python_exe, "venv", env_path
        except subprocess.CalledProcessError as e:
            print(f"Error creating venv: {e}", file=sys.stderr)
            return None, None, None

    if use_conda and _check_conda():
        return _create_conda()
    else:
        if use_conda:
            print("Conda not found. Falling back to standard Python venv.")
        return _create_venv()

def install_deps(python_exe: str):
    """Installs Python dependencies from require.txt."""
    requirements_path = os.path.join(SCRIPT_DIR, "require.txt")
    if not os.path.exists(requirements_path):
        print(f"Error: requirements file not found at {requirements_path}", file=sys.stderr)
        return False

    print("Installing Python dependencies from require.txt...")
    try:
        is_windows = sys.platform == "win32"
        subprocess.run(
            [python_exe, "-m", "pip", "install", "-r", requirements_path],
            check=True, shell=is_windows
        )
        print("Successfully installed Python dependencies.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e.stderr}", file=sys.stderr)
        return False

def install_rust_lib(python_exe: str, env_type: str, env_path: str):
    """Builds and installs the rust library into the specified Python environment."""
    print("\nBuilding and installing Rust library...")
    # We stay in the project root so maturin can find pyproject.toml
    custom_env = os.environ.copy()

    if env_type == 'conda':
        custom_env["CONDA_PREFIX"] = env_path
    elif env_type == 'venv':
        custom_env["VIRTUAL_ENV"] = env_path

    try:
        is_windows = sys.platform == "win32"
        # We run from the project root and let pyproject.toml guide maturin.
        # No more --manifest-path needed.
        subprocess.run(
            [python_exe, "-m", "maturin", "develop"],
            check=True, shell=is_windows, env=custom_env
        )
        print("Successfully installed Rust library.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Rust library: {e}", file=sys.stderr)

def validate_installation(python_exe: str, env_type: str, env_path: str):
    """
    Runs a quick Python command to validate that the core library was installed correctly.
    """
    print("\n--- Validating installation ---")
    os.chdir(PROJECT_ROOT)
    
    custom_env = os.environ.copy()
    if env_type == 'conda':
        custom_env["CONDA_PREFIX"] = env_path
    elif env_type == 'venv':
        custom_env["VIRTUAL_ENV"] = env_path
    custom_env["PYTHONIOENCODING"] = "utf-8"
        
    try:
        command = "from llm_core import Chat; print('Validation successful: llm_core.Chat imported.')"
        is_windows = sys.platform == "win32"
        subprocess.run(
            [python_exe, "-c", command], 
            check=True, capture_output=True, text=True, shell=is_windows,
            env=custom_env, encoding='utf-8'
        )
        return True
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] ❌ Validation failed.", file=sys.stderr)
        print("The Rust library could not be imported in the new environment.", file=sys.stderr)
        print("Please check the build logs above for errors.", file=sys.stderr)
        print(f"Error details: {e.stderr}", file=sys.stderr)
        return False

def remove_project_artifacts():
    """Removes all generated build files and the virtual environment."""
    print("--- Cleaning Project ---")
    
    items_to_remove = ["target", "__pycache__"]
    files_to_remove = ["Cargo.lock"]
    
    for root, dirs, files in os.walk(SCRIPT_DIR):
        # Remove directories
        for d in list(dirs):
            if d in items_to_remove or d.endswith(".egg-info"):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
        
        # Remove files
        for f in files:
            if f in files_to_remove:
                file_path = os.path.join(root, f)
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except OSError as e:
                    print(f"Could not remove file {file_path}: {e}", file=sys.stderr)

def remove_conda_env():
    # Attempt to remove the named Conda environment
    env_name = "llm_core_venv"
    print(f"Attempting to remove Conda environment '{env_name}'...")
    try:
        is_windows = sys.platform == "win32"
        subprocess.run(
            ["conda", "env", "remove", "-y", "--name", env_name],
            check=True, shell=is_windows, capture_output=True, text=True
        )
        print(f"Successfully removed Conda environment '{env_name}'.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Conda environment '{env_name}' not found or conda command failed. This is normal if you used venv.")

    # Also remove the local venv directory if it exists, as a fallback
    venv_path = get_env_dir()
    if os.path.exists(venv_path):
        print(f"Removing local venv environment: {venv_path}")
        shutil.rmtree(venv_path)

    print("--- Cleanup Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Manage the llm_core project setup.")
    parser.add_argument("-i", "--install", action="store_true", help="Create environment and install all dependencies.")
    parser.add_argument("-r", "--remove", action="store_true", help="Remove all build artifacts.")
    parser.add_argument("-t", "--total", action="store_true", help="Remove all build artifacts and the virtual environment.")
    parser.add_argument("--no-conda", action="store_true", help="Force the use of venv even if Conda is available.")
    args = parser.parse_args()

    env_name_str = 'llm_core'

    if args.install:
        print("--- Starting Project Installation ---")
        python_exe, env_type, env_path = create_env(use_conda=not args.no_conda, env_name_str=env_name_str)
        if not python_exe or not env_type or not env_path:
            print("Failed to create environment. Aborting.", file=sys.stderr)
            return

        if not install_deps(python_exe):
            print("Failed to install dependencies. Aborting.", file=sys.stderr)
            return
            
        # The logic to find rust projects is no longer needed.
        # pyproject.toml is the single source of truth.
        install_rust_lib(python_exe, env_type, env_path)
        
        if not validate_installation(python_exe, env_type, env_path):
            print("\nInstallation finished with errors. Please see validation output.", file=sys.stderr)
            return

        print("\n--- 🎉 Installation Complete! ---")
        print("\nThe environment is ready. To activate it for development, see the README.md in the core directory.")

    elif args.remove:
        remove_project_artifacts()
    elif args.total:
        remove_project_artifacts()
        remove_conda_env()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()