from dotenv import load_dotenv
import os

# import scripts from different files
from multislice import solve_multislice
from fds import solve_fds
from eval import compare

# Load the .env file
load_dotenv()

# Access variables
path = os.getenv("POT_PATH")

# Check if path loaded correctly
print(path)

# set params
# probably can use Settings class from given code
params = []

# get multislice result
result_ms = solve_multislice(params)

# get fds result
result_fds = solve_fds(params)

# compare result
result = compare(result_ms, result_fds)