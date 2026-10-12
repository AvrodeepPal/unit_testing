#unit_testing/python_test/test.py



import os
project_root = "python_test"
 
def run(project_root):
    file_list = []
    
    for dirpath, dirnames, filenames in os.walk(project_root):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Store path relative to project_root
            rel_path = os.path.relpath(filepath, project_root)
            file_list.append(rel_path)
            print(file_list)
 
    return file_list

def main():
    run("python_test")

if __name__ == "__main__":
    main()