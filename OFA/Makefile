.PHONY: docker-build docker-run-dev test

docker-build:
	
	docker build -t ofa -f Dockerfile .

docker-run:
	docker run -it --rm -v `pwd`:/home/ofa:rw --gpus all --network=host ofa