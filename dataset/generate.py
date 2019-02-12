import glob
import json

json_name = "keypoints.json"

def generate(image_name):
	template =  {
					"annotations": [],
					"class": "image",
					"filename": ""
				}
	template["filename"] = 	image_name		
	return template

def file_name():   
	return glob.glob("*.jpg")

if __name__ == "__main__":

	image_name = file_name()
	result = []
	for data in image_name:
		result.append(generate(data))

	with open(json_name, 'w') as f:
		json.dump(result, f)
	
	print("generate the config successful !")