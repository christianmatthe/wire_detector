# Import Darius' z-transfunctions
import socket
import time

class ZTranslator:
    """
    "The zTranslator class contains functions to adress the z-translator via 
    ip adress"
    
    """

    def __init__(self, ip="10.93.130.113", port=10001, steps_per_mm=40363.3, 
                 zero_position=11.71, max_range=35, velocity=0.24):
        self.ip = ip
        self.port = port
        self.steps_per_mm = steps_per_mm
        self.zero_position = zero_position
        self.velocity = velocity
        self.max_range = max_range
        self.create_connection()

    def create_connection(self):
        self.connection = socket.create_connection((self.ip, self.port), 
                                                   timeout=2)
        
    def close_connection(self):
        self.connection.close()

    def send(self, command):
        self.connection.send((command).encode())
        return self.connection.recv(2048).decode(errors='replace')
    
    def zero(self):
        """ Move to zero position. """
        print(f"zero: {0.0:.4f}")
        self.send(f" mr {int(-self.max_range*self.steps_per_mm)}\r")
        time.sleep(self.max_range/self.velocity)
        self.send(f" mr {int(self.zero_position*self.steps_per_mm)}\r")
        time.sleep(self.zero_position/self.velocity)
        self.current_position = 0

    def move_absolute(self, position):
        """ Move to absolute position in mm. Only works after zeroing. """
        if position > 20.5 or position < -11.5:
            raise ValueError("Position out of bounds.")
        if not hasattr(self, 'current_position'):
            raise ValueError("current_position not defined. Call zero() first.")
        print(f"move to position: {position:.4f}")
        if self.current_position != position:
            distance = position - self.current_position
            self.send(f" mr {int(distance*self.steps_per_mm)}\r")
            time.sleep(abs(distance)/self.velocity)
        self.current_position = position

    def move_relative(self, distance):
        """ Move a relative distance in mm. """
        print(f"move relative: {distance:.4f}")
        self.send(f" mr {int(distance*self.steps_per_mm)}\r")
        time.sleep(abs(distance)/self.velocity)
        if hasattr(self, 'current_position'):
            self.current_position += distance



####################################################################
if __name__ =="__main__": 
####################################################################
    # Enter sleeptime in seconds
    pre_sleep = 0 * 3600
    # Open connections (only one connection can be open at the same time)
    z_translator = ZTranslator()

    ####
    starttime = time.time()
    print(f'''Script started at {time.time()}; 
          sleeping for {pre_sleep/3600} hours''')
    #### sleep for {pre_sleep/3600} hour for thermalization and purifier 
    # prepping (done before)
    time.sleep(
                pre_sleep
                )

    #includes approximate slow controls delay
    z_time_interval = 2 * 3600 *(1+(0.78 + 0.2083)/500) 
    #z_time_interval = 60
    interval_time = starttime + z_time_interval

    # Shuffle positions to preclude time based trends
    pos_mm_list = [-11, -6, -1.5,  1,  3.5,
                    7, 12,  20,  -10, -5,
                    -1, 1.5, 4, 8, 13, 
                    18, -9, -4, -0.5, 2,
                    4.5, 9, 14, 19, -8,
                    -3, 0, 2.5, 5, 10,
                    15, 17, -7, -2, 0.5,
                    3, 6, 11, 16
                    ]
    # 39 positions * 2h = 78h

    ## Actual Runtime commands: 
    ##### Modified Copy of wire_z_trans_COM4_2h_dwell_2023-09-15.py
    z_translator.zero()
    current_pos_mm = 0
    for pos in pos_mm_list:
        move_mm = pos - current_pos_mm 
        z_translator.move_relative(move_mm)
        current_pos_mm = current_pos_mm + move_mm
        print(f"Stopped at {current_pos_mm} mm.")
        # wait till next measurement
        sleeptime = (z_time_interval 
                     - ((time.time() - starttime) % z_time_interval))
        print(f"Sleeping for {sleeptime} s.")
        time.sleep(
            sleeptime
            )
    # Leave z translator at 0
    z_translator.zero()