--
-- premake4 file to build TerrainRL
-- Copyright (c) 2009-2015 Glen Berseth
-- See license.txt for complete license.
--

local action = _ACTION or ""
local todir = "./" .. action

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local linuxLibraryLoc = "./external/"
local windowsLibraryLoc = "../library/"

solution "Example"

	configurations {"Release", "Debug"}
	configuration "Release"
		flags { "Optimize", "EnableSSE2","StaticRuntime", "NoMinimalRebuild", "FloatFast"}
	configuration "Debug"
		defines {"_DEBUG=1"}
		flags { "Symbols", "StaticRuntime" , "NoMinimalRebuild", "NoEditAndContinue"}


	platforms {"x32", "x64"}

	language "C++"

	
project "Example"
    -- prebuildcommands{
    --    projectDir .. "gen_swig.sh " .. projectDir;
    -- }

    language "C++"
	kind "SharedLib"
	
	targetdir ( "./" )
	-- The underscore here is IMPORTANT for the python naming import to work
	targetname ("_example")
	targetprefix ("")
	targetextension (".pyd")
	
	includedirs {
		"./include",
		"/usr/include/python2.7/",
		-- "/scratch/users/gberseth/modules/Python-2.7.3/Include/",	
		-- "/users/installs/python-2.7.3/include/python2.7/",
		"/users/installs/python-2.7.3/include/python2.7/",
		"C:/WinPython/python-3.5.2.amd64/include"
	}

	links {
		"python3"	
	}

	libdirs { 
			-- "/users/installs/python-2.7.3/lib/",
			"/scratch/users/gberseth/modules/Python-2.7.3/",
			"C:/WinPython/python-3.5.2.amd64/libs"
		}


	files {
		"include/*.h",
		"src/*.cpp",
		"example_wrap.cpp"
	}

	-- linux library cflags and libs
	configuration { "linux", "gmake" }
		buildoptions("-fPIC -ggdb" )
		
	-- windows library cflags and libs
	configuration { "windows" }
		-- On windows this needs to be the file extension for the library
		-- buildoptions("/IMPLIB _example.lib" )
		
	-- mac includes and libs
	configuration { "macosx" }
