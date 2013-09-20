// CIS565 CUDA Raytracer: A series of functions to simply manipulate the camera views in more human readable terms

#ifndef CAMERAUTILS_H
#define CAMERAUTILS_H

#include "sceneStructs.h"
#include "glm/glm.hpp"

//Orientation commands
void camLookUp(camera* cam, float degrees);
void camLookDown(camera* cam, float degrees);
void camLookRight(camera* cam, float degrees);
void camLookLeft(camera* cam, float degrees);
void camTiltLeft(camera* cam, float degrees);
void camTiltRight(camera* cam, float degrees);

//Movement commands
void camMoveForward(camera* cam, float units);



#endif