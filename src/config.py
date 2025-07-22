import os

base_dir = r"C:\Users\ACER\Desktop\New folder (2)"
exclude_folder = "rag"

with open("full_project_dump.txt", "w", encoding="utf-8") as output:
    for root, dirs, files in os.walk(base_dir):
        # Modify dirs in-place to skip 'rag'
        dirs[:] = [d for d in dirs if d != exclude_folder]

        for file in files:
            if file.endswith(".py"):  # include other file types if needed
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                output.write(f"\n\n# === {rel_path} ===\n\n")
                with open(file_path, "r", encoding="utf-8") as f:
                    output.write(f.read())
2048