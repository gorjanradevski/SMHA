import subprocess


def append_git_hash_on_file_path(file_path: str):
    """It gets the current git commit hash, and it appends it on the name, before the
    file extension.

    Args:
        file_path: The file path.

    Returns:
        The file path with the git commit hash appended on the name.

    """
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("utf-8")
        .rstrip()
    )
    path, extension = file_path.split(".")

    return path + "_" + git_hash + "." + extension
