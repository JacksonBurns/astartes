import re
import fileinput
import os

print(r'''
         .______    __           ___      .__   __.  __  ___
         |   _  \  |  |         /   \     |  \ |  | |  |/  /
         |  |_)  | |  |        /  ^  \    |   \|  | |  '  /
         |   _  <  |  |       /  /_\  \   |  . `  | |    <
         |  |_)  | |  `----. /  _____  \  |  |\   | |  .  \
         |______/  |_______|/__/     \__\ |__| \__| |__|\__\

   .______   ____    ____ .___________. __    __    ______   .__   __.
   |   _  \  \   \  /   / |           ||  |  |  |  /  __  \  |  \ |  |
   |  |_)  |  \   \/   /  `---|  |----`|  |__|  | |  |  |  | |   \|  |
   |   ___/    \_    _/       |  |     |   __   | |  |  |  | |  . `  |
   |  |          |  |         |  |     |  |  |  | |  `--'  | |  |\   |
   | _|          |__|         |__|     |__|  |__|  \______/  |__| \__|

.______   .______        ______          __   _______   ______ .___________.
|   _  \  |   _  \      /  __  \        |  | |   ____| /      ||           |
|  |_)  | |  |_)  |    |  |  |  |       |  | |  |__   |  ,----'`---|  |----`
|   ___/  |      /     |  |  |  | .--.  |  | |   __|  |  |         |  |
|  |      |  |\  \----.|  `--'  | |  `--'  | |  |____ |  `----.    |  |
| _|      | _| `._____| \______/   \______/  |_______| \______|    |__|

                            by Jackson Burns
               github.com/JacksonBurns/blank-python-project
''')

gh_uname = input("GitHub username: ")
while not re.search(r"^[a-z\d](?:[a-z\d]|-(?=[a-z\d])){0,38}$", gh_uname, re.IGNORECASE):
    print('''
Username may only contain alphanumeric characters or hyphens,
cannot have multiple consecutive hyphens, and cannot begin or end with a hyphen.
Maximum length is 39 characters.
''')
    gh_uname = input("GitHub username: ")

usr_name = input("Your name: ")

prj_name = input("Name of the project: ")
while not re.search(r"^[a-z\d](?:[a-z\d]|-|_(?=[a-z\d])){0,61}$", prj_name, re.IGNORECASE):
    print('''
Project name may only contain alphanumeric characters or hyphens/underscores,
cannot have multiple consecutive hyphens, and cannot begin or end with a hyphen.
Maximum length is 62 characters.
''')
    prj_name = input("Name of the project: ")

pypi_name = 'test pypi_name'  # input("Name for the PyPI package: ")

prj_name = input("Name for the PyPI package: ")
while not re.search(r"^[a-z\d](?:[a-z\d]|-|_(?=[a-z\d])){0,61}$", prj_name):
    print('''
    Package name may only contain lowercase alphanumeric characters or underscores and should be succinct.
    ''')
    prj_name = input("Name for the PyPI package: ")

slogan = input("Slogan for your project: ")

project_files = [
    'setup.py',
    'README.md',
    'test/test_blankpythonproject.py',
    'blankpythonproject/__init__.py',
    'blankpythonproject/blankpythonproject.py',
    '.github/workflows/run_unix_tests.yml',
    'docs/conf.py',
    'docs/index.rst',
    'docs/modules.rst',
]


def replace_blanks():
    changed = False
    for filename in project_files:
        with fileinput.FileInput(filename, inplace=True) as file:
            for line in file:
                if re.search("blankpythonproject", line):
                    print(line.replace('blankpythonproject', prj_name), end='')
                    changed = True
                elif re.search("blank-python-project", line):
                    print(line.replace('blank-python-project', prj_name), end='')
                    changed = True
                elif re.search("JacksonBurns", line):
                    print(line.replace('JacksonBurns', gh_uname), end='')
                    changed = True
                elif re.search("Jackson Burns", line):
                    print(line.replace('Jackson Burns', usr_name), end='')
                    changed = True
                elif re.search("blpyproj", line):
                    print(line.replace('blpyproj', pypi_name), end='')
                    changed = True
                elif re.search("Catchy slogan.", line):
                    print(line.replace('Catchy slogan.', slogan), end='')
                    changed = True
                else:
                    print(line, end='')
    return changed


it_limit = 0
while it_limit < 5:
    if not replace_blanks():
        break
    it_limit += 1

og_dir = os.getcwd()

os.rename('blankpythonproject_logo.png', prj_name + '_logo.png')

os.rename(
    os.path.join(og_dir, 'test', 'test_blankpythonproject.py'),
    os.path.join(og_dir, 'test', 'test_' + prj_name + '.py'),
)

os.rename(
    os.path.join(og_dir, 'examples', 'blankpythonproject_example.ipynb'),
    os.path.join(og_dir, 'examples', prj_name + '_example.ipynb'),
)

os.rename(
    os.path.join(og_dir, 'blankpythonproject', 'blankpythonproject.py'),
    os.path.join(og_dir, 'blankpythonproject', prj_name + '.py'),
)

os.rename('blankpythonproject', prj_name)
