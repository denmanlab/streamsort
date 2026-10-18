# Third-party components

streamsort is licensed under the GNU General Public License v3.0 (see `LICENSE`).

| component | where | license |
|---|---|---|
| Open Ephys processor-plugin template (structure of `OpenEphysLib.cpp`, `CMakeLists.txt`) | `plugin/streamsort-oe/` | GPL-3.0 (open-ephys-plugins/processor-plugin-template) |
| Open Ephys GUI headers (compile-time only, not included) | build dependency of `plugin/streamsort-oe/` | GPL-3.0 (open-ephys/plugin-GUI) |
| libzmq binaries and headers (copied from open-ephys-plugins/zmq-interface) | `plugin/streamsort-oe/libs/` | MPL-2.0 (compatible with GPL-3.0) |
| pybind11 (build dependency, not included) | `cpp/python/` | BSD-3-Clause |

Validation results and figures were derived from DANDI dataset 000774.
