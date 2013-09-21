-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Submitted 9/20/2013
-------------------------------------------------------------------------------
![This scene contains elements of every feature implemented](/renders/refractionwithmirrors.0.bmp "Hall of Mirrors")

Youtube Video of Rendering:
<dl>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=RdW1kkUx1jg
" target="_blank"><img src="http://img.youtube.com/vi/RdW1kkUx1jg/0.jpg" 
alt="Youtube Video of Rendering Process" width="480" height="360" border="10" /></a>
</dl>
-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! Any card
after the Geforce 8xxx series will work.

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This project is a completely functional ray tracing engine implemented in CUDA.  Ray Tracing is a technique in computer graphics that generates images by backtracing rays of light through pixels on the screen into a scene. By approximating various physical effects, rays can be reflected, refracted, and shaded by various lighting sources. Depending on the complexity of the scene and the enabled features images will be rendered in real time to <1 FPS. At any rate, it's still a lot faster than a CPU implementation. More information about Ray Tracing can be found here:
http://en.wikipedia.org/wiki/Ray_tracing_(graphics).

Ultimately this project will go on to become a full fledged path tracer that solves a more general form of the same rendering problem using stochastic sampling.

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project1 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio
  solution and the OSX and Linux makefiles reference this folder for all 
  source; the base source code compiles on Linux, OSX and Windows without 
  modification. (Note: the OSX and Linux versions have not been tested recently, use at your own risk)
* scenes/ contains an example scene description file.
* renders/ contains an example render of the given example scene file. 
* PROJ1_WIN/ contains a Windows Visual Studio 2010 project and all dependencies
  needed for building and running on Windows 7.
* PROJ1_OSX/ contains a OSX makefile, run script, and all dependencies needed
  for building and running on Mac OSX 10.8. 
* PROJ1_NIX/ contains a Linux makefile for building and running on Ubuntu 
  12.04 LTS. Note that you will need to set the following environment
  variables: 
    
  - PATH=$PATH:/usr/local/cuda-5.5/bin
  - LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/lib

  you may set these any way that you like.
    

The Windows and OSX versions of the project build and run exactly the same way
as in Project0.

-------------------------------------------------------------------------------
Features:
-------------------------------------------------------------------------------
The ray tracing engine implements several features:

* Raycasting from a camera into a scene through a pixel grid
* Phong lighting for multiple point light sources
* Diffuse surface rendering
* Raytraced soft shadows/area lighting
* Frame filtered progressive rendering
* Colored mirror reflection
* Simple refraction model
* Specular reflection 
* Supersampled adaptive antialiasing
* Minimally interactive camera

-------------------------------------------------------------------------------
Interactive Controls
-------------------------------------------------------------------------------
The engine was designed so that many features could modified at runtime to allow easy exploration of the effects of various parameters. In addition, several debug modes were implemented that graphically display additional information about the scene. These options to result in more complex kernels that have a negative impact on performance. I preferred the flexibility to quickly experiment for this project, but in the path tracer I will be redesigning the kernel structure from the ground up with performance in mind.

Here is a complete list of the keypress commands you can use at runtime.

Keypress | Function
--- | ---
A | Toggles Anti-Aliasing
S | Toggles soft shadows
x | Toggles adaptive shadows
F | Toggles Frame Filtering
f | Clears frame filter
] | Increase number of soft shadow rays
[ | Decrease number of soft shadow rays
= | Increase trace depth
- | Decrease trace depth
w/a/s/d | Move Camera Forward/Left/Back/Right
ESC | Exit
1 | Raytracing Render Mode
2 | Distance Debug Mode
3 | Normals Debug Mode
4 | Anti-aliasing Mode
5 | Shadow Debug Mode


-------------------------------------------------------------------------------
Debug Modes
-------------------------------------------------------------------------------
Distance debug mode casts rays and shades the distance to the first intersected surface in greyscale.
![Debug Mode](/screenshots/distance_debug.bmp "Distance Debug Mode")

Normal debug mode colors the normals of the first impacted surface for each ray. Pure RGB colors are axis aligned.
(i.e. Red pixels have normals along the x-axis)
![Debug Mode](/screenshots/normal_debug.bmp "Normal Debug Mode")

Shadow Debug Mode shows the pure light available at each pixel. There are some bugs here that have not been resolved, but still interesting.
![Debug Mode](/screenshots/shadow_debug.bmp "Shadow Debug Mode")

Aliasing Debug Mode highlights in green areas that are being adaptively oversampled.
![Debug Mode](/screenshots/aliasing_debug.PNG "Aliasing Debug Mode")


-------------------------------------------------------------------------------
Additional Screenshots
-------------------------------------------------------------------------------
![Screenshot](/renders/framefiltered.0.bmp "Example Scene with Frame Filtering")

![Screenshot](/renders/hallofmirrors.0.bmp "Hall of mirrors")

![Screenshot](/renders/hallofmirrorsdeep.0.bmp "Hall of mirrors deep trace")

![Screenshot](/renders/multisource.0.bmp "Multiple light sources, hard shadows")

![Screenshot](/renders/combined.0.bmp "Multiple light sources, soft shadows")

![Screenshot](/renders/combined.1.bmp "Multiple light sources, soft shadows")


-------------------------------------------------------------------------------
Blooper Reel
-------------------------------------------------------------------------------
Some floating point errors I encountered while implementing shadows.
![Screenshot](/renders/badshadow.0.bmp "Floating Point Errors")

Strange Refraction Artifact
![Screenshot](/renders/refraction.0.bmp "Refraction errors")



-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
As mentioned above, this code is not yet optimized for performance. 
However, I was able to experiment with various settings to get a feel for their impact on performance.

All of the following tests were run on the same machine running a GTX 760 with 1152 CUDA Cores. 
The same scene with 3 spheres and a single white spherical light in a box was used for all tests.

A baseline frame rate with Phong Shading, specular reflection, hard shadows, no AA, and no frame filtering was measured to be 20fps.

So what happens when we add soft shadows?

Number of Shadow Feelers | Frame Rate
--- | ---
1 | 20fps
2 | 15fps
3 | 13fps
4 | 11.5fps
5 | 10fps
6 | 10fps
7 | 8.5fps
10| 7fps

Clearly the number of shadow feelers negatively affects performance. 
However, from an aesthetic point of view adding frame filtering creates the same quality soft shadows with a single feeler while maintaining a 19-20fps frame rate.

-------------------------------------------------------------------------------
TAKUAscene FORMAT:
-------------------------------------------------------------------------------
This project uses a custom scene description format, called TAKUAscene.
TAKUAscene files are flat text files that describe all geometry, materials,
lights, cameras, render settings, and animation frames inside of the scene.
Items in the format are delimited by new lines, and comments can be added at
the end of each line preceded with a double-slash.

Materials are defined in the following fashion:

* MATERIAL (material ID)								//material header
* RGB (float r) (float g) (float b)					//diffuse color
* SPECX (float specx)									//specular exponent
* SPECRGB (float r) (float g) (float b)				//specular color
* REFL (bool refl)									//reflectivity flag, 0 for
  no, 1 for yes
* REFR (bool refr)									//refractivity flag, 0 for
  no, 1 for yes
* REFRIOR (float ior)									//index of refraction
  for Fresnel effects
* SCATTER (float scatter)								//scatter flag, 0 for
  no, 1 for yes
* ABSCOEFF (float r) (float b) (float g)				//absorption
  coefficient for scattering
* RSCTCOEFF (float rsctcoeff)							//reduced scattering
  coefficient
* EMITTANCE (float emittance)							//the emittance of the
  material. Anything >0 makes the material a light source.

Cameras are defined in the following fashion:

* CAMERA 												//camera header
* RES (float x) (float y)								//resolution
* FOVY (float fovy)										//vertical field of
  view half-angle. the horizonal angle is calculated from this and the
  reslution
* ITERATIONS (float interations)							//how many
  iterations to refine the image, only relevant for supersampled antialiasing,
  depth of field, area lights, and other distributed raytracing applications
* FILE (string filename)									//file to output
  render to upon completion
* frame (frame number)									//start of a frame
* EYE (float x) (float y) (float z)						//camera's position in
  worldspace
* VIEW (float x) (float y) (float z)						//camera's view
  direction
* UP (float x) (float y) (float z)						//camera's up vector

Objects are defined in the following fashion:
* OBJECT (object ID)										//object header
* (cube OR sphere OR mesh)								//type of object, can
  be either "cube", "sphere", or "mesh". Note that cubes and spheres are unit
  sized and centered at the origin.
* material (material ID)									//material to
  assign this object
* frame (frame number)									//start of a frame
* TRANS (float transx) (float transy) (float transz)		//translation
* ROTAT (float rotationx) (float rotationy) (float rotationz)		//rotation
* SCALE (float scalex) (float scaley) (float scalez)		//scale

An example TAKUAscene file setting up two frames inside of a Cornell Box can be
found in the scenes/ directory.

