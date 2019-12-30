rd /s build

rem set MSBuild="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
set MSBuild="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
rem call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
meson build --backend vs2019 --buildtype release ^
-Dcudnn_libdirs="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64","D:\cudnn-10.1-windows10-x64-v7.6.3.30\cuda\lib\x64" ^
-Dcudnn_include="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include","D:\cudnn-10.1-windows10-x64-v7.6.3.30\cuda\include" ^
-Ddefault_library=static

pause


cd build

%MSBuild%  ^
/p:Configuration=Release ^
/p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 lc0.sln ^
/filelogger

