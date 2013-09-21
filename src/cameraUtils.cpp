#include "cameraUtils.h"


void camMoveForward(camera* cam, float units)
{
	cam->positions[0] += cam->views[0]*units;
}


void camMoveRight(camera* cam, float units)
{
	glm::vec3 right = glm::cross(cam->views[0], cam->ups[0]);
	cam->positions[0] += right*units;
}