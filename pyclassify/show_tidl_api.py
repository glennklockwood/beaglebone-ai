#!/usr/bin/env python3
"""Shows the contents of the TIDL Python API.  Intended to be run on BeagleBone
AI's OS.
"""

import sys

sys.path.append("/usr/share/ti/tidl/tidl_api")

import tidl

print("TIDL version {}".format(tidl.__version__), end="\n\n")
print("{:28s} | {}\n{}|{}".format("Interface", "Type", "-" * 29, "-" * 49))

classes = []
for member_name in dir(tidl):
    if member_name.startswith("__"):
        continue
    member_name = ".".join([tidl.__name__, member_name])
    member = eval(member_name)
    member_type = str(type(member)).split()[-1].lstrip("'").rstrip(">'")
    print("{:27s} | {}".format(member_name, member_type))

    if member_type == "pybind11_builtins.pybind11_type":
        classes.append(member)

for clas in classes:
    print("-" * 80)
    print(clas)
    print("-" * 80)
    for member_name in dir(clas):
        if member_name.startswith("__"):
            continue
        member_name = ".".join([clas.__module__, clas.__name__, member_name])
        member = eval(member_name)
        member_type = str(type(member)).split()[-1].lstrip("'").rstrip(">'")
        print("{:50s} | {}".format(member_name, member_type))
    print()
