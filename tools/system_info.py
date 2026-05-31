from datetime import datetime


def print_system_info():
    """Collects information about the system and prints it to the console."""
    info = {}
    get_repo_info(info)
    get_platform_info(info)
    get_python_info(info)
    get_package_info_all(info)
    get_ffmpeg_info(info)
    print_info_dict(info)


def get_repo_info(info):
    try:
        # Requires package "gitpython"
        from git import Repo
        repo = Repo("..")
        repo_name = repo.remotes.origin.url.split("/")[-1]
        commit_hash = repo.head.object.hexsha[0:8]
        commit_date = repo.head.object.committed_date
        fmt = "%Y-%m-%d %H:%M:%S"
        commit_date = datetime.fromtimestamp(commit_date).strftime(fmt)
        changed_files = [item.a_path for item in repo.index.diff(None)]
    except Exception as e:
        repo_name = "<N/A>"
        commit_hash = "<N/A>"
        commit_date = "<N/A>"
        changed_files = "<N/A>"

    info["git_repo"] = repo_name
    info["git_commit_hash"] = commit_hash
    info["git_commit_date"] = commit_date
    info["changed_files"] = changed_files


def get_python_info(info):
    import sys
    info["python"] = sys.version.split(" ")[0]
    info["python_info"] = sys.version
    info["python_executable"] = sys.executable


def get_platform_info(info):
    import platform
    info["os"] = platform.system()
    if platform.system() == "Darwin":
        os_type = "Mac"
    elif platform.system() == "Windows":
        os_type = "Win"
    elif platform.system() == "Linux":
        os_type = "Linux"
    else:
        os_type = "Unknown"    
    info["os_type"] = os_type
    info["os_info"] = platform.platform()
    info["os_arch"] = platform.architecture()[0]


def get_package_info(info, name, package_name=None):
    if package_name is None:
        package_name = name
    if name != package_name:
        name = f"{name} ({package_name})"
    try:
        import importlib
        package = importlib.import_module(package_name)
        if hasattr(package, "__version__"):
            info["packages"][name] = package.__version__    
        else:
            info["packages"][name] = "available"
    except Exception as e:
        info["packages"][package_name] = "<N/A>"
        print(e)


def get_package_info_all(info):
    info["packages"] = {}
    get_package_info(info, "numpy")
    get_package_info(info, "scipy")
    get_package_info(info, "pandas")
    get_package_info(info, "matplotlib")
    get_package_info(info, "seaborn")

    get_package_info(info, "OpenCV", "cv2")
    get_package_info(info, "Pillow", "PIL")
    get_package_info(info, "skimage")

    get_package_info(info, "soundfile")
    get_package_info(info, "sounddevice")
    get_package_info(info, "jupyter")


def get_ffmpeg_info(info):
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE)
        version = result.stdout.decode("utf-8").split("\n")[0]
        version = version.split(" ")[2]
        info["ffmpeg"] = version
    except Exception as e:
        info["ffmpeg"] = "<N/A>"

    try:
        import ffmpeg
        info["ffmpeg_python"] = "available"
    except Exception as e:
        info["ffmpeg_python"] = "<N/A>"
        

def print_info_dict(info):
    print("Repository Info:")
    print("================")
    print("  {0:>14} : {1}".format("Name", info["git_repo"]))
    print("  {0:>14} : {1}".format("Commit", info["git_commit_hash"]))
    print("  {0:>14} : {1}".format("Date", info["git_commit_date"]))
    files = info["changed_files"]
    n = 3
    file_groups = [files[i:i+n] for i in range(0, len(files), n)]
    files = ("\n"+" "*21).join((", ".join(group) for group in file_groups))
    print("  {0:>14} : [ {1} ]".format("Edited files", files))
    print()
    print("Python:")
    print("=======")
    print("  {0:>14} : {1}".format("Version", info["python"]))
    print("  {0:>14} : {1}".format("Executable", info["python_executable"]))
    print()
        
    print("Platform Info:")
    print("==============")
    print("{0:>14} : {1} ({2})".format("OS", info["os_type"], info["os"]))
    print("{0:>14} : {1}".format("OS Info", info["os_info"]))
    print("{0:>14} : {1}".format("OS Arch", info["os_arch"]))
    print()

    print("Packages:")
    print("=========")
    max_len = max(len(k) for k in info["packages"])
    for k, v in info["packages"].items():
        v = v.lstrip("v")  # Remove prefix "v" if it occurs: "v0.1.2"
        print("{0:>{1}} : {2}".format(k, max_len, v))

    print()
    print("FFmpeg:")
    print("=======")
    print("{0:>14} : {1}".format("ffmpeg", info["ffmpeg"]))
    print("{0:>14} : {1}".format("ffmpeg-python", info["ffmpeg_python"]))



