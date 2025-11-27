
dcompose:
	sudo docker compose up

dclean:
	sudo docker rm haystack-rag-server-1
	sudo docker rmi haystack-rag-server
