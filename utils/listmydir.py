import os

# ============================================================
ROOT = "/mnt/netdisk2/evo2_explore/temp/nju-china-main-dev-0323/scripts/"
MAX_FILES = 3  
# ============================================================

def print_tree(root, prefix=""):
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        print(prefix + "  [无权限]")
        return

    # 分成子文件夹和文件两组
    dirs  = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]

    # 文件超出限制时截断，只显示前 MAX_FILES 个
    hidden = 0
    if len(files) > MAX_FILES:
        hidden = len(files) - MAX_FILES
        files  = files[:MAX_FILES]

    all_shown = dirs + files
    total = len(all_shown) + (1 if hidden else 0)

    for i, name in enumerate(all_shown):
        is_last = (i == total - 1) and (hidden == 0)
        connector = "└── " if is_last else "├── "
        print(prefix + connector + name)

        full_path = os.path.join(root, name)
        if os.path.isdir(full_path):
            extension = "    " if is_last else "│   "
            print_tree(full_path, prefix + extension)

    if hidden:
        print(prefix + f"└── ... ({hidden} more files)")


if __name__ == "__main__":
    print(ROOT)
    print_tree(ROOT)
