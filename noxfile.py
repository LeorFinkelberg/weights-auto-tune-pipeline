import nox


PYTHON_VERSION_DEFAULT = "3.11"
VENV_BACKEND = "uv"

nox.needs_version = ">=2023.04.22"
nox.options.default_venv_backend = VENV_BACKEND
nox.options.verbose = True


@nox.session(
    python=[PYTHON_VERSION_DEFAULT], venv_backend=VENV_BACKEND, reuse_venv=True
)
def tests(session):
    session.install("uv", "sync")
    session.run("pytest")
