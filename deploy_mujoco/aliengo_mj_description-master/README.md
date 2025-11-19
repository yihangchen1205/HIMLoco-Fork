# aliengo_mj_description
Aliengo robot model files for MuJoCo

The package will install the files in this directory so that [mc_mujoco](https://github.com/rohanpsingh/mc_mujoco) can pick them up automatically.

Install
-------

```bash
mkdir build
cd build
cmake ../
make && sudo make install
```

CMake options
-------------

- `SRC_MODE` if `ON` the files loaded by mujoco will point to the source rather than the installed files (default `OFF`)


If you want to use with `mc-rtc`, you will also need to install the corresponding RobotModule: [mc_aliengo](https://github.com/rohanpsingh/mc_aliengo).
