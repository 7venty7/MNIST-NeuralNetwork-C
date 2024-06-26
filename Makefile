CC = gcc
PRIMARY = neural_network
SECONDARY = utils/util
FLAGS = -lm -fsanitize=address

train:
	${CC} ${PRIMARY}.c ${SECONDARY}.c -o train ${FLAGS}
