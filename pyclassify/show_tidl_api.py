#!/usr/bin/env python3
"""Shows the contents of the TIDL Python API.  Intended to be run on BeagleBone
AI's OS.
"""

import sys

sys.path.append("/usr/share/ti/tidl/tidl_api")

import tidl

print("TIDL version {}".format(tidl.__version__), end="\n\n")
print("{:28s} | {}\n{}|{}".format("Interface", "Type", "-" * 29, "-" * 49))
for part in dir(tidl):
    print_cmd = "str(type(tidl.{}))".format(part)
    print("tidl.{:23s} | {}".format(
        part,
        str(eval(print_cmd)).split()[-1].lstrip("'").rstrip(">'")))
