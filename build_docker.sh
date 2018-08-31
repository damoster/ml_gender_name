if [ $# -eq 0 ]; then
	echo "Usage: sh $0 <image_name>"
else
	docker build -t $1 .
	echo "Docker image '$1' has been built, to start the container, run the following:\ndocker run -p 5000:5000 "$1	
fi