# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-src"
  "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-build"
  "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-subbuild/stb-populate-prefix"
  "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-subbuild/stb-populate-prefix/tmp"
  "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp"
  "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-subbuild/stb-populate-prefix/src"
  "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/yamamouni/Desktop/GPGPU/TP02/build/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
