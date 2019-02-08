#pragma once
// Std. Includes
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>




class Blendshape
{
public:
	std::vector<Vertex> baseModel;
	std::vector<std::vector<Vertex>> blendshapes;
	/*Functions */
	//Define the Blenshape for a given base shape and a path for the first blenshapes
	Blendshape(GLchar* path, std::vector<Vertex> baseModel) {
		this->baseModel = baseModel;
		this->loadModel(path);
	}
	//Define Blendshape with a base shapes
	Blendshape(std::vector<Vertex> baseModel) {
		this->baseModel = baseModel;
	}
	//Add a new model to the blendshape model
	void addShape(GLchar* path) {
		this->loadModel(path);
	}
	void addShape(std::vector<Vertex> blend) {
		this->loadModel(blend);
	}
	//Get the number of Vertices used in the blendshape 
	int getNumberVertices(int i) {
		return blendshapes[i].size();
	}
	//Function to retrieve a specific blendshape model
	std::vector<Vertex> getBlendShape(int i) {
		return blendshapes[i];
	}
private:
	/* Model Data */
	std::string directory;
	/* Functions */
	//Load Model from file using Assimp Importer
	void loadModel(std::string path)
	{
		Assimp::Importer import;
		const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

		if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
			std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
			return;
		}
		this->directory = path.substr(0, path.find_last_of('/'));

		this->processNode(scene->mRootNode, scene);

	}
	void processNode(aiNode* node, const aiScene* scene) {
		// Process all the node's meshes (if any)
		for (GLuint i = 0; i < node->mNumMeshes; i++) {
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			this->blendshapes.push_back(this->processMesh(mesh, scene));
		}

		for (GLuint i = 0; i < node->mNumChildren; i++) {
			this->processNode(node->mChildren[i], scene);
		}
	}
	//Define a Model from another
	void loadModel(std::vector<Vertex> blend) {

		//Remove the base model from the model given 
		for (unsigned int i = 0; i < blend.size(); i++) {
			glm::vec3 vector;
			vector.x = blend[i].Position.x - baseModel[i].Position.x;
			vector.x = blend[i].Position.y - baseModel[i].Position.y;
			vector.x = blend[i].Position.z - baseModel[i].Position.z;
			blend[i].Position = vector;
		}

		blendshapes.push_back(blend);
	}
	std::vector<Vertex> processMesh(aiMesh* mesh, const aiScene* scene)
	{
		assert(baseModel.size() == mesh->mNumVertices);
		std::vector<Vertex> vertices;

		for (GLuint i = 0; i < mesh->mNumVertices; i++) {
			Vertex vertex;
			// Process vertex positions
			// Define the vertices as the change from the mean
			glm::vec3 vector;
			vector.x = mesh->mVertices[i].x - baseModel[i].Position.x;
			vector.y = mesh->mVertices[i].y - baseModel[i].Position.y;
			vector.z = mesh->mVertices[i].z - baseModel[i].Position.z;
			vertex.Position = vector;
			//Normals
			vector.x = mesh->mNormals[i].x;
			vector.y = mesh->mNormals[i].y;
			vector.z = mesh->mNormals[i].z;
			vertex.Normal = vector;
			//Texture
			if (mesh->mTextureCoords[0]) {
				glm::vec2 vec;
				vec.x = mesh->mTextureCoords[0][i].x;
				vec.y = mesh->mTextureCoords[0][i].y;
				vertex.TexCoords = vec;
			}
			else
				vertex.TexCoords = glm::vec2(0.0f, 0.0f);

			vertices.push_back(vertex);

		}

		return vertices;
	}
	
};