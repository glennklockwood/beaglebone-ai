This repository contains code from my experimentation with BeagleBone AI and
the TIDL API.  Of note,

`classification/` is a cleaned-up version of [TI's ImageNet classification
example implemented in C++][1].

`pyclassify/` is a reimplementation of the [BeagleBone AI TIDL classification
demo][2] from the [cloud9-examples][] repo that uses the TIDL Python API
and Flask.  I find this easier to understand since it is a standalone Python
script instead of an mjpg-streamer plugin library.

This is all a work-in-progress.

[1]: https://git.ti.com/cgit/tidl/tidl-api/tree/examples/imagenet/main.cpp
[2]: https://github.com/beagleboard/cloud9-examples/blob/v2020.01/BeagleBone/AI/tidl/classification.tidl.cpp
[cloud9-examples]: https://github.com/beagleboard/cloud9-examples
