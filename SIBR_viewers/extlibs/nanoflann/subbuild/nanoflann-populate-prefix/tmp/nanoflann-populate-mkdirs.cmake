# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/nanoflann"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/build"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/tmp"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
