## IN THIS PROGRAM, A USER INPUTS THE LABEL ONCE AND THEN KNOCKS HOWEVER MANY TIMES THEY WOULD LIKE
import json
import serial

# Function to find sequence windows
def process_acceleration_readings(output, temp_output):
    print("test")

# Function to save the acceleration readings
def save_acceleration_readings():
    print("SAVING")
    data = []
    for i in range(len(knock_sequence_data)):
        data.append({
            "sequence": knock_sequence_data[i],
            "label": LABEL
        })
    # Open a file for writing
    with open("data.json", "w") as f:
        # Write the data to the file as a JSON object
        json.dump(data, f)

# Open a connection to the serial port
ser = serial.Serial('COM5', baudrate=19200)

# Create a temporary list that contains the 3 most recent samples so that the first sample is always 2 samples before the threshold is broken
temp_output = [0, 0, 0]

WINDOW_LENGTH = 200 # 1000 sample window length
THRESHOLD = 80 # threshold value of 80

knock_sequence_data = [] # list of acceleration readings -> [sequence number][sample number] -> accel. data
current_sequence_data = []
#knock_label_data = [] # list of labels associated with a sequence -> [sequence number] -> label

# Ask the user to enter their name
LABEL = input("Please enter your name: ")

sequence_counter = 0 # counter for current sequence number, each new knock sequence iterates counter
sample_counter = 0 # counter for samples for starting and ending sequence window

# Read and print the serial data from the Arduino
while (len(knock_sequence_data) < 10):
    ##Take in serial port, clean data, and format to int
    line = ser.readline()  # read a byte
    if line:
        string = line.decode()  # convert the byte string to a unicode string
        string = string.strip()
        try:
            output = float(string)
            # if Window is already initted update sequence to include most recent output
            if (sample_counter > 0):
                current_sequence_data.append(output)
                sample_counter += 1
                print(current_sequence_data)
            else:
                temp_output[2] = temp_output[1]  # store temp outputs with index 0 being most recent
                temp_output[1] = temp_output[0]
                temp_output[0] = output
                if (output > THRESHOLD):
                    current_sequence_data.append(temp_output[2])
                    current_sequence_data.append(temp_output[1])
                    current_sequence_data.append(temp_output[0])
                    sample_counter = 3

            if (sample_counter >= WINDOW_LENGTH):
                knock_sequence_data.append(current_sequence_data.copy())
                for list in knock_sequence_data:
                    print(list)
                current_sequence_data.clear()
                sample_counter = 0
                sequence_counter += 1

        except ValueError: print("VALUE ERROR")

ser.close()

save_acceleration_readings()