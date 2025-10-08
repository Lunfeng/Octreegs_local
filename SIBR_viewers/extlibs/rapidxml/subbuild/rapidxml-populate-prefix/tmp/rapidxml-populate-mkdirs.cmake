# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/rapidxml"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/build"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/tmp"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src"
  "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/Documents/Python Projects/Octree-GS/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
