CC = gcc
PRIMARY = neural_network
SECONDARY = utils/util
FLAGS = -lm -fsanitize=address -g -Wall

train:
	${CC} ${PRIMARY}.c ${SECONDARY}.c -o train ${FLAGS}
