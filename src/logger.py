# it have all the record 
# it track all the things
import logging 
import os  
from datetime import datetime  # Import datetime to generate timestamps

# Create log file name with current timestamp (format: MM_DD_YYYY_HH_MM_SS)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create full path for logs directory with the log file name
logs_path = os.path.join(os.getcwd(),"logs" , LOG_FILE)

# Create logs directory if it doesn't exist (exist_ok=True prevents error if dir exists)
os.makedirs(logs_path , exist_ok=True)

# Create the complete file path where logs will be stored
LOG_FILE_PATH = os.path.join(logs_path , LOG_FILE)

# Configure basic logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Specify the log file path

    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Format: timestamp, line number, logger name, level, message
    
    level= logging.INFO,  # Set logging level to INFO (captures INFO, WARNING, ERROR, CRITICAL)
)

# # Main execution block
# if __name__  == "__main__":
#     logging.info("Logging has started")  # Log initial message