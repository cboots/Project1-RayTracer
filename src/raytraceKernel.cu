// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
#include <helper_math.h>
#else
#include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
		exit(EXIT_FAILURE); 
	}
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Function that traverses scene searching for collisions. Traces ray to first impact. Returns index of first geometry hit or -1 if no collision
__host__ __device__ int firstIntersect(staticGeom* geoms, int numberOfGeoms, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, float& distance)
{
	//Index of the first hit geometry
	int firstGeomInd = -1;
	distance = -1;
	//Best intersection points stored in output params as minimums encountered. Limits temporary variables..

	//for each geometry object
	//TODO create better scene graph to improve collision detection for more complicated scenes. (Octtree)
	for(int i = 0; i < numberOfGeoms; ++i)
	{
		//Temporary return variables
		glm::vec3 intersectionPointTemp;
		glm::vec3 normalTemp;

		//Test for collision
		float dist = geomIntersectionTest(geoms[i], r, intersectionPointTemp, normalTemp);
		if(dist > 0.0)
		{
			//Impact detected
			if(distance < 0 || dist < distance)
			{
				//First hit or closer hit
				distance = dist;
				firstGeomInd = i;
				intersectionPoint = intersectionPointTemp;
				normal = normalTemp;
			}
		}
	}

	return firstGeomInd;
}

__host__ __device__ glm::vec3 reflect(glm::vec3 incident, glm::vec3 normal)
{
	return incident-glm::dot(2.0f*normal, incident) * normal;
}

///Compute the scalar contribution from specular highlights
__host__ __device__ float calculateSpecularScalar(ray viewRay, ray lightDirection, glm::vec3 normal, float specularExponent)
{
	float dot = glm::dot(reflect(-lightDirection.direction, normal), -viewRay.direction);
	if(dot <= 0.0)
		return 0.0;
	else
		return glm::pow(dot, specularExponent);
		
	
}

///Compute the scalar contribution from diffuse lighting
__host__ __device__ float calculateDiffuseScalar(glm::vec3 normal, ray lightDirection)
{
	return MAX(0,glm::dot(normal, lightDirection.direction));
}

__host__ __device__ glm::vec3 computeShadowedIntensity(ray primeRay, glm::vec3 intersectionPoint, int hitGeomIndex, glm::vec3 normal, renderOptions rconfig, 
													   int time,	staticGeom* geoms, int numberOfGeoms, material* mats, int numberOfMaterials, int lightIndex, ray& lightDirection)
{
	//TODO: implement shadows
	material lightMat =  mats[geoms[lightIndex].materialid];

	//For now just return the center of the light source. Treat as a point source
	lightDirection.origin = intersectionPoint;
	lightDirection.direction = glm::normalize(geoms[lightIndex].translation - intersectionPoint);
	return lightMat.color*lightMat.emittance;
}

//Computes the illumination contributions from each light source at this point.
//Incorporates ambient, specular, and diffuse reflection as well as shadows.
//Returns the summed light intensity in rgb components. Perfect reflection and refraction effects are not included.
__host__ __device__ glm::vec3 calculatePhongIllumination(ray primeRay, glm::vec3 intersectionPoint, int hitGeomIndex, glm::vec3 normal, renderOptions rconfig, 
														 int time,	staticGeom* geoms, int numberOfGeoms, material* mats, int numberOfMaterials)
{

	//Initialize to ambient component.
	glm::vec3 totalL = rconfig.ka*rconfig.ambientLight*mats[geoms[hitGeomIndex].materialid].color;

	//for each material, if it's a light source add its acumulated 
	//TODO: precompute which objects are light sources
	for(int i = 0; i < numberOfGeoms; ++i){
		if(mats[geoms[i].materialid].emittance > 0)
		{
			if(i == hitGeomIndex)
			{
				//we hit a light, add in its own component
			}
			//Light source, compute contribution
			ray lightDirection;//An output variable that returns the direction to the center of the effective light source
			glm::vec3 lightIntensity = computeShadowedIntensity(primeRay, intersectionPoint, hitGeomIndex, normal, rconfig, 
				time,	geoms, numberOfGeoms, mats, numberOfMaterials, i, lightDirection);

			if(lightIntensity.x > 0 || lightIntensity.y > 0 || lightIntensity.z > 0){
				//Compute diffuse contribution
				if(rconfig.kd > 0)
				{

					//kd is a global tuning parameter that allows control of each lighting element.
					totalL += rconfig.kd*(lightIntensity*mats[geoms[hitGeomIndex].materialid].color)
						*calculateDiffuseScalar(normal, lightDirection);
				}

				//Compute Specular contribution
				if(rconfig.ks > 0 && mats[geoms[hitGeomIndex].materialid].specularExponent > 0)
				{
					totalL += rconfig.ks*(lightIntensity*mats[geoms[hitGeomIndex].materialid].specularColor)
						*calculateSpecularScalar(primeRay, lightDirection, normal, mats[geoms[hitGeomIndex].materialid].specularExponent);
					
				}
			}
		}
	}

	return totalL;
}


//TODO: verify raycastFromCameraKernel FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, 
												glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
	ray r;
	r.origin = eye;
	glm::vec3 right = glm::cross(view, up);

	//float d = 1.0f; use a viewing plane of 1 distance 
	glm::vec3 pixel_location = /* d* */(view + (2*x/resolution.x-1)*right*glm::tan(glm::radians(fov.x)) 
		- (2*y/resolution.y-1)*up*glm::tan(glm::radians(fov.y)));

	r.direction = glm::normalize(pixel_location);

	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		image[index] = glm::vec3(0,0,0);
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;
		color.x = image[index].x*255.0;
		color.y = image[index].y*255.0;
		color.z = image[index].z*255.0;

		if(color.x>255){
			color.x = 255;
		}

		if(color.y>255){
			color.y = 255;
		}

		if(color.z>255){
			color.z = 255;
		}

		// Each thread writes one pixel location in the texture (textel)
		PBOpos[index].w = 0;
		PBOpos[index].x = color.x;
		PBOpos[index].y = color.y;
		PBOpos[index].z = color.z;
	}
}

//TODO: IMPLEMENT raytraceRay Kernel FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, renderOptions rconfig, glm::vec3* colors,
							staticGeom* geoms, int numberOfGeoms, material* mats, int numberOfMaterials)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y)){  
		//Valid pixel, away we go!
		//First we must have a primary ray
		ray primeRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);

		//Calculate impact of primary ray
		float dist;
		glm::vec3 intersectionPoint;
		glm::vec3 normal;
		int ind = firstIntersect(geoms, numberOfGeoms, primeRay, intersectionPoint, normal, dist);

		if(ind >= 0)
		{
			//We have something to draw. 
			switch(rconfig.mode)
			{
			case NORMAL_DEBUG:
				//Debug render. Display normals of very first impacted surface.
				colors[index] = glm::abs(normal);
				break;
			case DISTANCE_DEBUG:
				colors[index] = glm::vec3(1,1,1)*(1-dist/rconfig.distanceShadeRange);
				break;
			case RAYTRACE:
				//TODO Implement actual raytracer here
				//colors[index] = mats[geoms[ind].materialid].color;
				colors[index] = calculatePhongIllumination(primeRay, intersectionPoint, ind, normal, rconfig, time, 
					geoms, numberOfGeoms, mats, numberOfMaterials);


				break;
			}
		}else{
			//Clear pixels that don't hit anything
			colors[index] = glm::vec3(0,0,0);
		}
	}
}

//TODO: FINISH Kernel Wrapper FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, renderOptions* renderOpts, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), 
		(int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	//send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	//package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[i] = newStaticGeom;
	}

	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	material* cudamats = NULL;
	cudaMalloc((void**)&cudamats, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamats, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);


	//package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;


	ray r;
	r.origin = glm::vec3(0,0,10);
	r.direction = glm::vec3(0,0,-1);
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	float result = boxIntersectionTest(geomList[0], r, intersectionPoint, normal);

	//kernel launches
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, *renderOpts, cudaimage, cudageoms, numberOfGeoms, cudamats, numberOfMaterials);

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	//free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudamats  );
	cudaFree( cudageoms );
	delete geomList;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
