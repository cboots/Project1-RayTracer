#include "cameraUtils.h"


void camMoveForward(camera* cam, float units)
{
	cam->positions[0] += cam->views[0]*units;
}
