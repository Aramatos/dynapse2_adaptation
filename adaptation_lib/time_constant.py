import time
import numpy as np
from dwf import constants, DWF


def record_analog(duration, frequency=1000):
    # Initialize the device
    dwf = DWF()

    # Open the first device
    dwf.open(0)

    # Set up acquisition
    dwf.analogInFrequencySet(frequency)
    buffer_size = frequency * duration
    if buffer_size > dwf.analogInBufferSizeMax(): # Check if the buffer size exceeds the maximum
        print("Requested buffer size exceeds the maximum of {}. Reducing to maximum.".format(dwf.analogInBufferSizeMax()))
        buffer_size = dwf.analogInBufferSizeMax() 
    dwf.analogInBufferSizeSet(buffer_size)
    dwf.analogInChannelEnableSet(0, True) 
    dwf.analogInChannelRangeSet(0, 5) 

    # Wait for the configuration to stabilize
    time.sleep(0.1)

    # Start the other function in a new thread
    thread = threading.Thread(target=other_function)
    thread.start()

    # Start acquisition
    dwf.analogInConfigure(False, True)

    print("Recording for {} seconds...".format(duration))
    time.sleep(duration)  # Wait for the required duration

    # Fetch the data
    data = dwf.analogInStatus(True, True)
    values = dwf.analogInStatusData(0, buffer_size)  # Fetch data for channel 1

    # Create timestamps for each sample
    timestamps = np.arange(0, duration, 1.0/frequency)

    # Terminate the other function
    thread.do_run = False
    thread.join()

    # Close the device
    dwf.close()

    # Return the timestamps and recorded values
    return timestamps[:len(values)], values

# Now call the function
duration = 10  # Record for 10 seconds
timestamps, data = record_analog(duration)
