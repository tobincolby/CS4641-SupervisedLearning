import numpy as np

car_file = open("data/original_car_quality.csv")
car_data = np.loadtxt(car_file, delimiter=",")

student_file = open("data/tic-tac-toe_original.csv")
student_data = np.loadtxt(student_file, delimiter=",")

np.random.shuffle(car_data)
np.random.shuffle(student_data)


car_training, car_test = car_data[:int(len(car_data)*0.7),:], car_data[int(len(car_data)*0.7):,:]
student_training, student_test = student_data[:int(len(student_data)*0.7),:], student_data[int(len(student_data)*0.7):,:]

np.savetxt("data/car_quality_train.csv", car_training, delimiter=",")
np.savetxt("data/tic_train.csv", student_training, delimiter=",")

np.savetxt("data/car_quality_test.csv", car_test, delimiter=",")
np.savetxt("data/tic_test.csv", student_test, delimiter=",")

