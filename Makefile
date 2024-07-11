CC = gcc
PRIMARY = train
MATRIX = matrix/matrix
NNET = nnet/nnet
FLAGS = -lm -fsanitize=address -g -Wall

train:
	${CC} ${PRIMARY}.c ${NNET}.c ${MATRIX}.c -o train ${FLAGS}
