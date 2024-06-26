CC = gcc
PRIMARY = neural_network
SECONDARY = utils/util
FLAGS = -lm -lutils

train: ${PRIMARY}
	${CC} ${PRIMARY}.c ${SECONDARY}.c ${FLAGS} -o train
