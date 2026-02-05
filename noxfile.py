import nox


PYTHON_VERSION_DEFAULT = "3.11"
VENV_BACKEND = "uv"

nox.options.sessions = ["lint"]
nox.needs_version = ">=2023.04.22"
nox.options.verbose = True
nox.options.default_venv_backend = VENV_BACKEND
nox.options.reuse_existing_virtualenvs = True


@nox.session(
    python=[PYTHON_VERSION_DEFAULT], venv_backend=VENV_BACKEND, reuse_venv=True
)
def lint(session):
    session.run("uv", "sync")
    session.run("pre-commit", "run", "--all-files")
