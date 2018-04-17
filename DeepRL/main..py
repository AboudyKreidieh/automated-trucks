import socket

HOST = 'localhost'  # Symbolic name meaning all available interfaces
PORT = 50005  # Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

print("waiting for response from client at port {}...".format(PORT))
conn, address = s.accept()
print('Connected by', address)

HORIZON = 1500

time_counter = 0
while True:
    # receive new state data
    data = conn.recv(1024)

    if data:
        # convert data to string
        str_data = data.decode('ascii')

        # increment the time counter
        time_counter += 1

        # process the data
        (veh1_pos, veh1_vel, veh1_accel, veh2_vel, veh2_vel, veh2_accel) = \
            tuple(str_data.split(":"))

        # compute the next action from these variables
        # TODO

        # send new actions back to Matlab
        # TODO
        # conn.sendall(data)

        if time_counter >= HORIZON:
            conn.sendall(b" 1.00: 1.00:1\n")
            # enough steps were run; stop trying to connect to the server
            break

        conn.sendall(b" 1.00: 1.00:0\n")

        # close the connection
        conn.close()

        # wait for the next connection
        conn, address = s.accept()

# close the connection
conn.close()
